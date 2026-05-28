"""
arXiv AI Research Digest
Fetches recent papers, scores relevance with Claude, and emails a daily digest.
"""

import os
import json
import base64
import html
import re
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
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
MAX_SUMMARIES = 10

# Model to use. Swapping to "claude-haiku-4-5-20251001" cuts cost ~10x
# at the expense of summary quality — good option if you want to run this
# multiple times per day.
CLAUDE_MODEL = "claude-opus-4-7"

# Gmail OAuth will only request permission to SEND mail — not read, not delete.
# This is the narrowest scope that does the job.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

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
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    search = arxiv.Search(
        query=query,
        max_results=MAX_PAPERS_FETCHED,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Anything submitted before this moment is too old for today's digest.
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    papers = []
    for result in client.results(search):
        # Results arrive newest-first. The moment one is older than the cutoff,
        # every paper after it is older too — so we can stop early.
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

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SCORING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Score these {len(papers)} papers:\n\n{paper_block}",
        }],
    )

    raw = response.content[0].text.strip()

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

# ===========================================================================
# >>>>>>>>>>   PLACEHOLDER PROMPT — WRITE YOUR REAL ONE HERE LATER   <<<<<<<<<<
# ===========================================================================
# This is a stub. Replace the text below with your own summarization prompt
# whenever you're ready. Just keep the {title} and {abstract} placeholders so
# each paper's data gets filled in.
SUMMARY_PROMPT_TEMPLATE = """Summarize this arXiv paper in exactly 3 bullet points:
- What it does
- What's new
- Why it matters

Title: {title}
Abstract: {abstract}"""
# ===========================================================================


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

def render_email_html(summaries):
    """Build the HTML email body from the list of summarized papers.

    Every piece of paper text is HTML-escaped before it goes into the page,
    because titles/abstracts/summaries come from outside our code (arXiv +
    Claude). Escaping stops a stray '<' or '<script>' from breaking — or
    injecting into — the email. Same trust-boundary habit as before.
    """
    today = datetime.now().strftime("%B %d, %Y")

    parts = [f"""\
<div style="font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; max-width: 680px; margin: 0 auto; color: #1a1a1a; line-height: 1.5;">
  <h1 style="font-size: 20px; border-bottom: 2px solid #b31b1b; padding-bottom: 8px;">arXiv AI Digest &middot; {today}</h1>
  <p style="color: #666; font-size: 14px;">{len(summaries)} paper(s) cleared today's relevance bar.</p>"""]

    for p in summaries:
        title = html.escape(p["title"])
        abs_url = html.escape(p["abstract_url"])
        category = html.escape(p["primary_category"])
        score = int(p.get("score", 0))
        hook = html.escape(p.get("hook", ""))
        summary_html = html.escape(p.get("summary", "")).replace("\n", "<br>")

        # Show the first three authors, then "et al." so the meta line stays short.
        authors = p.get("authors", [])
        author_str = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "")
        author_str = html.escape(author_str)

        # pdf_url can be None (withdrawn papers), so only add the PDF link if present.
        pdf_url = p.get("pdf_url")
        pdf_link = f' &middot; <a href="{html.escape(pdf_url)}">PDF</a>' if pdf_url else ""

        parts.append(f"""\
  <div style="margin: 22px 0; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px;">
    <h2 style="font-size: 17px; margin: 0 0 6px;"><a href="{abs_url}" style="color: #b31b1b; text-decoration: none;">{title}</a></h2>
    <div style="font-size: 13px; color: #666; margin-bottom: 10px;">
      <span style="background: #b31b1b; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{score}/10</span>
      &nbsp; {category} &nbsp;&middot;&nbsp; {author_str}
    </div>
    <p style="font-style: italic; color: #444; margin: 10px 0;">{hook}</p>
    <div style="font-size: 14px; margin: 10px 0;">{summary_html}</div>
    <div style="font-size: 13px;"><a href="{abs_url}">abstract</a>{pdf_link}</div>
  </div>""")

    parts.append("</div>")
    return "\n".join(parts)
