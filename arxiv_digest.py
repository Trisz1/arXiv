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
