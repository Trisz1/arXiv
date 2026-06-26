"""
arXiv AI Research Digest
Fetches recent papers, scores relevance with Claude, and emails a daily digest.
"""

import os
import json
import base64
import html
import re
import time
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText

import arxiv
import anthropic
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.AR", "cs.DC", "cs.NE"]
LOOKBACK_HOURS = 36
MAX_PAPERS_FETCHED = 200
RELEVANCE_THRESHOLD = 7
MAX_SUMMARIES = 5

# Model to use. Swapping to "claude-haiku-4-5-20251001" cuts cost ~10x
# at the expense of summary quality — good option if you want to run this
# multiple times per day.
CLAUDE_MODEL = "claude-opus-4-7"

# Gmail OAuth will only request permission to SEND mail — not read, not delete.
# This is the narrowest scope that does the job.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# .strip() guards against a trailing newline or space sneaking in when these get
# pasted into env vars / GitHub secrets. A newline inside a header value (the API
# key, or the "To" address) is rejected by HTTP as an "illegal header value".
ANTHROPIC_API_KEY = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
RECIPIENT_EMAIL = (os.environ.get("RECIPIENT_EMAIL") or "").strip()

# ---------------------------------------------------------------------------
# INTEREST PROFILE
# ---------------------------------------------------------------------------
# This is what Claude reads when deciding if a paper is relevant to you.
# Edit freely — the more specific, the better the filtering.

INTEREST_PROFILE = """
I am interested in papers that fall into any of the following areas:

HIGH INTEREST (score 8-10):
- AI infrastructure and compute systems (chips, memory, networking for ML)
- Inference efficiency: faster/cheaper ways to run large models at scale
- Training systems: distributed training, parallelism strategies, memory optimisation
- Foundation models: architecture innovations in LLMs, multimodal models
- Agent architectures: multi-agent systems, tool use, planning, memory
- Compute economics: cost curves, scaling laws, the business of running AI
- Anything with clear commercial or startup implications in the AI stack

MEDIUM INTEREST (score 5-7):
- Reinforcement learning with practical real-world applications
- AI safety and alignment (especially technical, not just philosophical)
- New benchmarks that reveal something genuinely new about model capabilities
- Open-source model releases with architectural novelty

LOW INTEREST / SKIP (score 1-4):
- Narrow computer vision or audio-only work with no broader AI relevance
- Incremental benchmark improvements on existing tasks (+1% on GLUE, etc.)
- Highly theoretical proofs without near-term practical implications
- Domain-specific applications (medical imaging, satellite imagery, etc.)
  unless the underlying technique is broadly novel
"""


# ---------------------------------------------------------------------------
# SECTION 2: FETCH PAPERS FROM ARXIV
# ---------------------------------------------------------------------------

def fetch_recent_papers():
    """Pull recent papers from arXiv across our categories.

    Returns a list of dicts (newest first), each holding the fields we need
    downstream: title, abstract, authors, category, and links.
    """
    # Build a query matching ANY of our categories: "cat:cs.AI OR cat:cs.LG OR ..."
    query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)

    # The client handles paging and politeness: it waits a few seconds between
    # pages so we don't hammer arXiv's free public API (and get rate-limited).
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)

    search = arxiv.Search(
        query=query,
        max_results=MAX_PAPERS_FETCHED,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Anything submitted before this moment is too old for today's digest.
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    # arXiv rate-limits shared/cloud IPs (like GitHub Actions runners) with HTTP
    # 429. The limit usually clears after a short wait, so retry the whole fetch
    # with exponential backoff (10s, 20s, 40s, 80s) before giving up.
    last_error = None
    for attempt in range(5):
        try:
            papers = []
            for result in client.results(search):
                # Results arrive newest-first. The moment one is older than the
                # cutoff, every paper after it is older too — so stop early.
                if result.published < cutoff:
                    break
                papers.append({
                    "title": " ".join(result.title.split()),
                    "abstract": " ".join(result.summary.split()),
                    "authors": [a.name for a in result.authors],
                    "primary_category": result.primary_category,
                    "abstract_url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "published": result.published,
                })
            return papers
        except arxiv.HTTPError as e:
            last_error = e
            if attempt < 4:
                wait = 10 * (2 ** attempt)
                print(f"  arXiv error ({e}); retrying in {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)

    raise SystemExit(f"arXiv kept rate-limiting us after 5 tries. Last error: {last_error}")


# ---------------------------------------------------------------------------
# SECTION 3: SCORE RELEVANCE (one batched Claude call)
# ---------------------------------------------------------------------------

# Standing instructions for the scoring call. This is written at column 0 (no
# indentation) so the text Claude receives is exactly what you see here.
SCORING_SYSTEM_PROMPT = f"""You are a research assistant filtering arXiv papers for one specific reader.

Here is the reader's interest profile:
{INTEREST_PROFILE}

You will receive a numbered list of papers, each with an id, title, primary
category, and abstract. For EVERY paper, assign:
  - "score": an integer from 1 to 10 (10 = perfect fit, 1 = irrelevant)
  - "hook": ONE sentence (max ~25 words) saying why it might matter to this
    reader. Be specific and concrete, not generic praise.

Output ONLY a JSON array — one object per paper — exactly like this:
[{{"id": 0, "score": 8, "hook": "..."}}, {{"id": 1, "score": 3, "hook": "..."}}]

Do not write anything before or after the JSON array. Do not wrap it in
markdown code fences."""


def score_relevance(client, papers):
    """Score every paper in ONE Claude call, then filter and sort.

    Returns the papers scoring >= RELEVANCE_THRESHOLD, highest first, each with
    "score" and "hook" added. Returns [] on any failure (an empty digest beats
    a crash).
    """
    if not papers:
        return []

    # Build a compact, numbered block. We send only what's needed to judge
    # relevance (id, title, category, abstract) — not authors or links — to
    # keep the single call as cheap as possible.
    paper_block = "\n\n".join(
        f"[{i}] Title: {p['title']}\n"
        f"Category: {p['primary_category']}\n"
        f"Abstract: {p['abstract']}"
        for i, p in enumerate(papers)
    )

    # Stream this call: scoring 200 papers generates thousands of tokens, which
    # holds the connection open for minutes. A non-streaming request that long
    # gets dropped on some networks (e.g. GitHub Actions runners). Streaming keeps
    # data flowing so the connection stays alive; get_final_message() then
    # collects the complete reply.
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SCORING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Score these {len(papers)} papers:\n\n{paper_block}",
        }],
    ) as stream:
        message = stream.get_final_message()

    raw = message.content[0].text.strip()

    # Defensive: we TOLD Claude to emit bare JSON, but we don't trust it blindly.
    # It sometimes wraps output in ```json ... ``` fences anyway — strip them.
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        scored = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ! Could not parse Claude's scoring JSON ({e}); skipping this run.")
        return []

    if not isinstance(scored, list):
        print("  ! Scoring response was not a JSON array; skipping this run.")
        return []

    # Map scores back onto the original papers by id. Coerce types defensively
    # because this is model output — never assume it's well-formed.
    keepers = []
    for item in scored:
        try:
            idx = int(item["id"])
            score = int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(papers) or score < RELEVANCE_THRESHOLD:
            continue
        paper = papers[idx]
        paper["score"] = score
        paper["hook"] = str(item.get("hook", "")).strip()
        keepers.append(paper)

    keepers.sort(key=lambda p: p["score"], reverse=True)
    return keepers


# ---------------------------------------------------------------------------
# SECTION 4: SUMMARIZE A SINGLE PAPER (one Claude call per paper)
# ---------------------------------------------------------------------------

# Your summarization prompt. Edit this any time to change how each paper is
# written up — just keep the {title} and {abstract} placeholders.
SUMMARY_PROMPT_TEMPLATE = """You are writing a daily AI research digest for a sharp, AI-fluent reader — picture a bright 22-year-old recent graduate who uses AI tools every day but has no engineering, machine-learning, or research background. Respect their intelligence: explain clearly and directly, never write down to them, and avoid cutesy analogies or childish comparisons.

Write exactly these three sections. Keep each label exactly as written, on its own line, with its text on the following line(s):

In simple terms:
2-3 sentences explaining what this is and why it matters. Explain the actual idea, not a hand-holding metaphor. You may assume the reader knows everyday AI concepts (large language models, chatbots, AI agents), but explain the underlying science or industry context they wouldn't know, defining any specialized term briefly in parentheses.

Where it could matter:
1 sentence on who might use this — the real-world situations and the kinds of companies it could be relevant to.

A little more technical:
2-3 sentences going a level deeper on what the paper does and what is genuinely new about it. You may use technical terms here; define anything specialized in parentheses.

Title: {title}
Abstract: {abstract}

Start directly with "In simple terms:" — no preamble."""


def summarize_paper(client, paper):
    """Generate a short summary for ONE paper via a single Claude call.

    Returns the summary text. If the call fails (rate limit, network blip), we
    return a fallback note instead of raising — one bad paper shouldn't sink
    the other nine summaries in the digest.
    """
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        title=paper["title"],
        abstract=paper["abstract"],
    )
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  ! Summary failed for '{paper['title'][:50]}...': {e}")
        return "(Summary unavailable for this paper.)"


# ---------------------------------------------------------------------------
# SECTION 5: RENDER THE EMAIL HTML
# ---------------------------------------------------------------------------

def _format_summary_html(summary):
    """Turn Claude's summary text into clean, friendly HTML.

    We escape the untrusted text FIRST, then apply only our own light formatting:
    a short line ending in ":" becomes a colored section heading, **x** becomes
    bold, and everything else becomes a readable paragraph. (Escape-before-format
    keeps the same injection protection as the rest of the renderer.)
    """
    pieces = []
    for line in summary.split("\n"):
        line = line.strip()
        if not line:
            continue
        safe = html.escape(line)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        if len(line) <= 45 and line.endswith(":"):
            # Section label: small uppercase in accent blue, no divider above —
            # the design rhythm comes from spacing, not lines.
            pieces.append(
                f'<div style="font-size: 11.5px; font-weight: 600; letter-spacing: 0.08em; '
                f'text-transform: uppercase; color: #0071e3; margin-bottom: 9px;">{safe[:-1]}</div>'
            )
        else:
            pieces.append(
                f'<p style="margin: 0 0 26px; font-size: 16px; line-height: 1.65; color: #3d3d40;">{safe}</p>'
            )
    return "\n".join(pieces)


def render_email_html(summaries):
    """Build a warm, easy-to-read HTML digest from the summarized papers.

    All paper text is HTML-escaped before it enters the page (titles, hooks, and
    summaries come from arXiv + Claude — untrusted), keeping the same injection
    protection as before.
    """
    today = datetime.now().strftime("%B %d, %Y")

    cards = []
    for p in summaries:
        title = html.escape(p["title"])
        abs_url = html.escape(p["abstract_url"])
        category = html.escape(p["primary_category"])
        score = int(p.get("score", 0))
        hook = html.escape(p.get("hook", ""))
        summary_html = _format_summary_html(p.get("summary", ""))

        # First three authors, then "et al." so the meta line stays short.
        authors = p.get("authors", [])
        author_str = html.escape(", ".join(authors[:3]) + (", et al." if len(authors) > 3 else ""))

        # pdf_url can be None (withdrawn papers); only show the PDF link if present.
        pdf_url = p.get("pdf_url")
        pdf_link = (f'<a href="{html.escape(pdf_url)}" '
                    f'style="font-weight: 500; color: #0071e3; text-decoration: none;">Download PDF</a>'
                    if pdf_url else "")

        cards.append(f"""\
  <div style="background: #ffffff; border-radius: 18px; padding: 38px 40px; margin: 22px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 12px 32px rgba(0,0,0,0.06); text-align: left;">
    <a href="{abs_url}" style="display: block; font-size: 21px; font-weight: 600; line-height: 1.28; letter-spacing: -0.015em; color: #1d1d1f; text-decoration: none; margin-bottom: 14px;">{title}</a>
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 18px;">
      <span style="font-size: 12px; font-weight: 600; color: #515154; background: #f0f0f2; border-radius: 999px; padding: 4px 11px;">Relevance {score}/10</span>
      <span style="font-size: 13.5px; color: #86868b;">{category} &middot; {author_str}</span>
    </div>
    <p style="font-style: italic; font-size: 15.5px; line-height: 1.55; color: #86868b; margin: 0 0 28px;">{hook}</p>
{summary_html}
    <div style="height: 1px; background: #ededef; margin: 0 0 18px;"></div>
    <div style="display: flex; gap: 18px; font-size: 14px;">
      <a href="{abs_url}" style="font-weight: 500; color: #0071e3; text-decoration: none;">Read abstract</a>{pdf_link}
    </div>
  </div>""")

    body = "\n".join(cards)
    return f"""\
<div style="background: #f5f5f7; padding: 8px 0 28px; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <div style="max-width: 660px; margin: 0 auto; padding: 0 16px;">
    <div style="padding: 36px 4px 14px;">
      <div style="font-size: 11.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #0071e3; margin-bottom: 10px;">Today&rsquo;s research</div>
      <div style="font-size: 28px; font-weight: 600; color: #1d1d1f; letter-spacing: -0.02em; line-height: 1.1;">Your AI Research Digest</div>
      <div style="font-size: 15px; color: #86868b; margin-top: 6px;">{today} &middot; {len(summaries)} paper(s)</div>
    </div>
{body}
    <div style="border-top: 1px solid #ededef; margin-top: 40px; padding-top: 28px; text-align: center; color: #86868b; font-size: 13.5px;">Built on open research from arXiv.org</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# SECTION 6: GMAIL (OAuth sign-in + send)
# ---------------------------------------------------------------------------

def get_gmail_service():
    """Authenticate with Gmail and return a service object for sending mail.

    - First ever run: opens a browser so you can click "Allow", then writes
      token.json.
    - Every run after: silently loads token.json (refreshing it if the short
      access token has expired) — no browser needed.
    """
    creds = None

    # token.json is your saved sign-in from a previous run.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", GMAIL_SCOPES)

    # No valid sign-in on hand? Either refresh it, or do the one-time browser flow.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # silent: swap the expired token for a fresh one
        else:
            # credentials.json identifies OUR app to Google; this opens the browser.
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the sign-in so future runs skip the browser.
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(service, to, subject, html_body):
    """Send one HTML email through the Gmail API."""
    # charset utf-8 so accented author names / unicode in titles don't crash encoding.
    message = MIMEText(html_body, "html", "utf-8")
    message["To"] = to
    message["Subject"] = subject

    # The Gmail API wants the raw RFC-822 message, base64url-encoded.
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()


# ---------------------------------------------------------------------------
# SECTION 7: MAIN (orchestration)
# ---------------------------------------------------------------------------

def main():
    # Fail fast, with a clear message, if the secrets aren't set.
    if not ANTHROPIC_API_KEY:
        raise SystemExit("ERROR: set the ANTHROPIC_API_KEY environment variable first.")
    if not RECIPIENT_EMAIL:
        raise SystemExit("ERROR: set the RECIPIENT_EMAIL environment variable first.")

    # max_retries lets the SDK ride out transient network blips and rate limits
    # (it retries connection errors / 429 / 5xx with exponential backoff).
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=5)

    print(f"Fetching papers from arXiv (last {LOOKBACK_HOURS}h, cap {MAX_PAPERS_FETCHED})...")
    papers = fetch_recent_papers()
    print(f"  -> {len(papers)} papers fetched.")
    if not papers:
        print("Nothing fetched. Exiting.")
        return

    print("Scoring relevance in one batched Claude call...")
    keepers = score_relevance(client, papers)
    print(f"  -> {len(keepers)} papers scored >= {RELEVANCE_THRESHOLD}.")
    if not keepers:
        print("No papers cleared the bar today. No email sent.")
        return

    # Hard cap for cost control: only summarize the top N.
    keepers = keepers[:MAX_SUMMARIES]
    print(f"Summarizing top {len(keepers)} (cap is {MAX_SUMMARIES})...")
    for i, paper in enumerate(keepers, 1):
        print(f"  [{i}/{len(keepers)}] {paper['title'][:60]}...")
        paper["summary"] = summarize_paper(client, paper)

    print("Rendering email...")
    html_body = render_email_html(keepers)

    print("Authenticating with Gmail (browser opens on first run only)...")
    service = get_gmail_service()

    today = datetime.now().strftime("%b %d, %Y")
    subject = f"arXiv AI Digest - {today} ({len(keepers)} papers)"
    print(f"Sending digest to {RECIPIENT_EMAIL}...")
    send_email(service, RECIPIENT_EMAIL, subject, html_body)
    print("Done! Digest sent.")


if __name__ == "__main__":
    main()
