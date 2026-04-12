"""
Job Scout: finds a company's careers page, scrapes job listings,
and ranks them by relevance to a candidate's resume.
"""
import json
import logging
import re
from typing import Optional
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CAREERS_PATHS = [
    "/careers",
    "/jobs",
    "/careers/jobs",
    "/about/careers",
    "/company/careers",
    "/en/careers",
    "/us/careers",
    "/work-with-us",
    "/join-us",
    "/open-positions",
    "/opportunities",
]

JOB_TITLE_RE = re.compile(
    r"(engineer|developer|scientist|analyst|manager|designer|architect|"
    r"lead|director|specialist|consultant|intern|researcher|product|"
    r"data|software|machine\s*learning|\bml\b|\bai\b|backend|front.?end|"
    r"fullstack|devops|cloud|platform|security|operations|quantitative|"
    r"applied|research|principal|staff|senior|junior)",
    re.IGNORECASE,
)


def _get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    """GET a URL; return Response on 200 with non-trivial body, else None."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 500:
            return resp
    except Exception as exc:
        logger.debug("GET %s failed: %s", url, exc)
    return None


def _slugify(company: str) -> str:
    """'Google LLC' → 'google', 'Amazon Web Services' → 'amazon'."""
    first = company.strip().split()[0]
    return re.sub(r"[^a-z0-9]", "", first.lower())


def find_careers_url(company_name: str) -> Optional[str]:
    """
    Try common URL patterns for a company's careers page.
    Falls back to a DuckDuckGo HTML search if none match.
    Returns the first working URL, or None if not found.
    """
    slug = _slugify(company_name)

    # Direct candidates
    candidates: list[str] = (
        [f"https://www.{slug}.com{p}" for p in CAREERS_PATHS]
        + [
            f"https://careers.{slug}.com",
            f"https://jobs.{slug}.com",
            f"https://{slug}.careers",
            # Known ATS sub-domains
            f"https://boards.greenhouse.io/{slug}",
            f"https://jobs.lever.co/{slug}",
        ]
    )

    for url in candidates:
        resp = _get(url)
        if resp:
            logger.info("Found careers page: %s", url)
            return url

    # DuckDuckGo HTML fallback (no API key needed)
    query = f"{company_name} official careers jobs site"
    search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    resp = _get(search_url, timeout=15)
    if resp:
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.select(".result__a"):
            href = a.get("href", "")
            text = a.get_text(strip=True).lower()
            if any(kw in (href + text) for kw in ["/careers", "/jobs", "careers.", "jobs."]):
                # DuckDuckGo wraps real URLs in a redirect param
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = unquote(m.group(1))
                if href.startswith("http"):
                    logger.info("Found careers page via DuckDuckGo: %s", href)
                    return href

    logger.warning("Could not find careers page for '%s'", company_name)
    return None


def scrape_job_listings(careers_url: str, max_jobs: int = 40) -> list[dict]:
    """
    Scrape job titles and links from a careers page.
    Returns a list of {title, url, snippet} dicts.
    """
    resp = _get(careers_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    parsed = urlparse(careers_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    jobs: list[dict] = []
    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        href: str = a["href"]

        # Filter by title length and job-title keyword
        if not title or len(title) < 6 or len(title) > 140:
            continue
        if not JOB_TITLE_RE.search(title):
            continue

        # Normalise URL
        if href.startswith("http"):
            full_url = href
        elif href.startswith("/"):
            full_url = base + href
        else:
            continue

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Grab surrounding text as a snippet
        snippet = ""
        parent = a.parent
        if parent:
            snippet = parent.get_text(separator=" ", strip=True)[:300]

        jobs.append({"title": title, "url": full_url, "snippet": snippet})
        if len(jobs) >= max_jobs:
            break

    return jobs


def rank_jobs(jobs: list[dict], resume_text: str, llm_service) -> list[dict]:
    """
    Ask the LLM to score each job title 0-100 for relevance to the resume.
    Sorts descending by score and attaches rank to each job.
    Falls back to score=50 for all if the LLM call fails.
    """
    if not jobs:
        return []

    job_lines = "\n".join(f"{i + 1}. {j['title']}" for i, j in enumerate(jobs))
    prompt = (
        "You are a career advisor. Score each job title's relevance (0-100) "
        "to the candidate's resume. Higher = more relevant.\n\n"
        f"RESUME (first 2500 chars):\n{resume_text[:2500]}\n\n"
        f"JOB TITLES:\n{job_lines}\n\n"
        f"Return ONLY a JSON array: "
        f'[{{"index":1,"score":82}}, ...] for all {len(jobs)} jobs. No other text.'
    )

    result = llm_service.generate_response(prompt, max_length=1024)

    if result.get("status") != "success":
        for i, j in enumerate(jobs):
            j.update({"score": 50, "rank": i + 1})
        return jobs

    try:
        text = result["response"].strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            rankings = json.loads(m.group())
            score_map = {
                r["index"]: r.get("score", 50)
                for r in rankings
                if isinstance(r, dict) and "index" in r
            }
            for i, job in enumerate(jobs):
                job["score"] = score_map.get(i + 1, 50)
            jobs.sort(key=lambda x: x.get("score", 50), reverse=True)
            for rank, job in enumerate(jobs, 1):
                job["rank"] = rank
    except Exception as exc:
        logger.warning("Ranking parse failed: %s", exc)
        for i, j in enumerate(jobs):
            j.update({"score": 50, "rank": i + 1})

    return jobs
