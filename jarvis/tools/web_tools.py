"""Web search and page fetch tools."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # type: ignore


async def web_search(query: str, brave_api_key: str = "", limit: int = 5) -> str:
    """Search the web. Uses Brave Search if api key provided, else DuckDuckGo."""
    if brave_api_key:
        return await _brave_search(query, brave_api_key, limit)
    return await _ddg_search(query, limit)


async def _brave_search(query: str, api_key: str, limit: int) -> str:
    import httpx

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": query, "count": limit}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("web", {}).get("results", [])
    lines = [f"{i+1}. {r['title']} — {r['url']}\n   {r.get('description','')}"
             for i, r in enumerate(results)]
    return "\n".join(lines) or "(no results)"


async def _ddg_search(query: str, limit: int) -> str:
    if DDGS is None:
        return "(duckduckgo_search not installed)"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=limit))
    lines = [f"{i+1}. {r['title']} — {r['href']}\n   {r['body']}"
             for i, r in enumerate(results)]
    return "\n".join(lines) or "(no results)"


async def fetch_page(url: str) -> str:
    """Fetch a URL and return clean extracted text."""
    import httpx
    import trafilatura

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 Jarvis/2.0"})
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html) or ""
    if not text.strip():
        return "(could not extract text from page)"
    return text[:8000]
