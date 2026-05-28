"""
arXiv AI Research Digest
Fetches recent papers, scores relevance with Claude, and emails a daily digest.
"""

import os
import json
import base64
import re
import textwrap
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
