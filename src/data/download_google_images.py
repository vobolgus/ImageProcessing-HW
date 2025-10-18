#!/usr/bin/env python3
"""
Download images from Google, Bing, and DuckDuckGo for a list of search queries.

Features:
- Processes a list of hardcoded queries (TARGET_QUERIES).
- Downloads a specified number of images per query (IMAGES_PER_QUERY).
- Creates a main run directory with a timestamp, and a separate subdirectory for each query.
- Multithreaded downloads with progress bars and error handling.
- Rotates User-Agents to reduce the chance of being blocked.

Notes:
- This script scrapes search engine HTML. It may break if the markup changes
  or if access is rate-limited. For robust, high-volume tasks, consider an official API.
- Ensure you have permissions to write to the 'data' directory.

Dependencies: requests, beautifulsoup4, tqdm

Run:
  pip install requests beautifulsoup4 tqdm
  python download_google_images.py
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import re
import threading
import time
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import quote_plus

# --------------------------- Configuration ---------------------------
# A list of queries to download images for.
TARGET_QUERIES = [
    "macbook pro 16 inch space gray",
    "macbook pro 14 inch silver",
    "macbook air m2 midnight",
    "macbook air m3 starlight",
    "macbook pro m1 max",
    "macbook air 15 inch",
    "macbook pro on desk",
    "macbook air lifestyle",
    "developer using macbook pro",
    "macbook pro closed lid",
    "macbook air open screen",
    "macbook pro keyboard",
    "macbook air side view",
    "macbook pro ports",
    "macbook pro 2023",
    "macbook air 2022",
    "macbook pro 13 inch",
    "silver macbook air",
    "space gray macbook air",
    "macbook pro touch bar",
    "macbook with external monitor",
    "macbook pro silver top view",
]

# Number of images to attempt to download for EACH query.
IMAGES_PER_QUERY = 100  # Reduced for faster testing; you can increase it back to 1000.
NUM_THREADS = 32  # Number of download threads

# Timeout settings (seconds)
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

# Pagination/collection controls
MAX_PAGES = 200  # Upper bound for pages to visit per search engine
EMPTY_PAGE_LIMIT = 35  # Stop after this many consecutive pages add no new URLs
DEFAULT_PAUSE = 0.3  # Seconds between page fetches

# Enable/disable search engines
ENABLE_GOOGLE = True
ENABLE_BING = False
ENABLE_DDG = False
MAX_PAGES_BING = 200
MAX_PAGES_DDG = 200

# User-Agents to rotate per request
USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
    ),
]
# The first agent is kept for backwards compatibility in functions that don't rotate.
USER_AGENT = USER_AGENTS[0]


# --------------------------------------------------------------------

@dataclass
class DownloadResult:
    index: int
    url: str
    ok: bool
    path: str | None
    error: str | None


def slugify(value: str) -> str:
    """
    Simple slugify: lowercase, replace spaces with underscores, and remove invalid chars.
    """
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]+", "", value)
    return value or "query"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_jpg_or_png_url(u: str) -> bool:
    """Accept common web image formats by extension."""
    return re.search(r"\.(?:jpe?g|png|webp|gif)(?:$|[?#])", u, flags=re.IGNORECASE) is not None


def is_google_ui_or_thumb(u: str) -> bool:
    """Filter out search engine UI assets and low-res thumbnails."""
    u_low = u.lower()
    return (
            "google.com/images/branding" in u_low
            or "encrypted-tbn" in u_low  # Google thumbnails
            or "gstatic.com/images" in u_low
            or "/favicon" in u_low
    )


def google_image_search_urls(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    """Fetch candidate image URLs from Google Images HTML pages."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    urls: List[str] = []
    seen: Set[str] = set()
    empty_pages = 0
    stop_reason: str | None = None
    pages_visited = 0

    q = quote_plus(query)
    with tqdm(total=needed, desc="Collecting URLs (Google)", unit="url", leave=False) as pbar:
        for ijn in range(MAX_PAGES):
            if len(urls) >= needed:
                stop_reason = "reached_needed"
                break

            pages_visited += 1
            ua = USER_AGENTS[ijn % len(USER_AGENTS)]
            headers_local = dict(headers)
            headers_local["User-Agent"] = ua

            # Alternate between pagination styles
            page_param = f"ijn={ijn}" if ijn % 2 == 0 else f"start={ijn * 50}"
            url = f"https://www.google.com/search?q={q}&tbm=isch&hl=en&{page_param}"

            try:
                resp = requests.get(url, headers=headers_local, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if resp.status_code != 200:
                    time.sleep(pause + random.uniform(0, 0.3))
                    continue
            except Exception:
                time.sleep(pause + random.uniform(0, 0.3))
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            prev_len = len(urls)

            # Strategy 1: Parse structured data from result grid
            for tag in soup.select("div.isv-r img[data-iurl], div.isv-r img[src]"):
                candidate = tag.get("data-iurl") or tag.get("src") or tag.get("data-src")
                if candidate and candidate.startswith("http") and not is_google_ui_or_thumb(
                        candidate) and is_jpg_or_png_url(candidate):
                    if candidate not in seen:
                        seen.add(candidate)
                        urls.append(candidate)
                        pbar.update(1)
                        if len(urls) >= needed: break
            if len(urls) >= needed: break

            # Strategy 2: Parse JSON "ou" (original URL) fields
            ou_matches = re.findall(r'"ou":"(https?://[^"]+)"', resp.text)
            for raw in ou_matches:
                u = raw.replace("\\u003d", "=").replace("\\u0026", "&").replace("\\u002F", "/")
                if not is_google_ui_or_thumb(u) and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    pbar.update(1)
                    if len(urls) >= needed: break
            if len(urls) >= needed: break

            if len(urls) == prev_len:
                empty_pages += 1
                if empty_pages >= EMPTY_PAGE_LIMIT:
                    stop_reason = "empty_page_limit"
                    break
            else:
                empty_pages = 0
            time.sleep(pause)

    if stop_reason is None:
        stop_reason = "max_pages_reached" if pages_visited >= MAX_PAGES else "loop_end"

    pbar.close()
    print(f"  [Google] Found {len(urls)} URLs. Stop reason: {stop_reason}.")
    return urls[:needed]


def collect_from_bing(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    """Fetch candidate URLs from Bing Images."""
    if needed <= 0: return []

    urls: List[str] = []
    seen: Set[str] = set()
    q = quote_plus(query)
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.bing.com/"}

    with tqdm(total=needed, desc="Collecting URLs (Bing)", unit="url", leave=False) as pbar:
        for page in range(MAX_PAGES_BING):
            if len(urls) >= needed: break
            first = page * 50 + 1
            url = f"https://www.bing.com/images/search?q={q}&first={first}&count=50&form=HDRSC2&adlt=off"
            try:
                resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if resp.status_code != 200:
                    time.sleep(pause + random.uniform(0.1, 0.4))
                    continue
            except Exception:
                time.sleep(pause + random.uniform(0.1, 0.4))
                continue

            for m in re.findall(r'"murl":"(https?://[^"\\]+)"', resp.text):
                u = m.replace("\\/", "/").replace("\\u0026", "&")
                if "mm.bing.net" not in u.lower() and "/th?id=" not in u.lower() and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    pbar.update(1)
                    if len(urls) >= needed: break
            time.sleep(pause + random.uniform(0, 0.3))

    pbar.close()
    print(f"  [Bing] Found {len(urls)} URLs.")
    return urls[:needed]


def collect_from_ddg(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    """Fetch candidate URLs from DuckDuckGo's JSON endpoint."""
    if needed <= 0: return []

    urls: List[str] = []
    seen: Set[str] = set()
    q = quote_plus(query)
    headers = {"User-Agent": USER_AGENT, "Referer": "https://duckduckgo.com/"}

    # Obtain vqd token
    try:
        init_resp = requests.get(f"https://duckduckgo.com/?q={q}&iax=images&ia=images", headers=headers,
                                 timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        vqd_match = re.search(r"vqd='([\w-]+)'", init_resp.text)
        if not vqd_match: return []
        vqd = vqd_match.group(1)
    except Exception:
        return []

    s = 0
    with tqdm(total=needed, desc="Collecting URLs (DDG)", unit="url", leave=False) as pbar:
        for _ in range(MAX_PAGES_DDG):
            if len(urls) >= needed: break
            api_url = f"https://duckduckgo.com/i.js?l=en-us&o=json&q={q}&vqd={vqd}&f=,&p=1&s={s}"
            try:
                api_resp = requests.get(api_url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if api_resp.status_code != 200: break
                data = api_resp.json()
            except Exception:
                break

            found_new = False
            for item in data.get("results", []):
                u = item.get("image")
                if u and not is_google_ui_or_thumb(u) and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    pbar.update(1)
                    found_new = True
                    if len(urls) >= needed: break

            if not found_new or "next" not in data: break
            s += len(data["results"])
            time.sleep(pause + random.uniform(0, 0.2))

    pbar.close()
    print(f"  [DDG] Found {len(urls)} URLs.")
    return urls[:needed]


def collect_image_urls(query: str, needed: int) -> List[str]:
    """Orchestrate URL collection from all enabled search engines."""
    all_urls: List[str] = []
    seen: Set[str] = set()

    def add_unique(urls_to_add: List[str]):
        for u in urls_to_add:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)

    if ENABLE_GOOGLE and len(all_urls) < needed:
        add_unique(google_image_search_urls(query, needed - len(all_urls)))

    if ENABLE_BING and len(all_urls) < needed:
        add_unique(collect_from_bing(query, needed - len(all_urls)))

    if ENABLE_DDG and len(all_urls) < needed:
        add_unique(collect_from_ddg(query, needed - len(all_urls)))

    print(f"  Total unique URLs collected for query: {len(all_urls)} (requested: {needed})")
    return all_urls[:needed]


def ext_from_response(original_url: str, resp: requests.Response) -> str:
    """Determine file extension from Content-Type header or the final URL after redirects."""
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "image/jpeg" in ct: return ".jpg"
    if "image/png" in ct: return ".png"
    if "image/webp" in ct: return ".webp"
    if "image/gif" in ct: return ".gif"

    # Fallback to checking the file extension in the final URL (after redirects)
    m = re.search(r"\.(jpe?g|png|webp|gif)(?:[?#]|$)", resp.url, flags=re.IGNORECASE)
    if m:
        return "." + m.group(1).lower()

    # As a last resort, check the original URL
    m_orig = re.search(r"\.(jpe?g|png|webp|gif)(?:[?#]|$)", original_url, flags=re.IGNORECASE)
    if m_orig:
        return "." + m_orig.group(1).lower()

    return ".jpg"  # Default fallback


def download_one(index: int, url: str, out_dir: Path, session: requests.Session) -> DownloadResult:
    """Download a single image file."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://www.google.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        r = session.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True,
                        allow_redirects=True)
        if r.status_code != 200:
            return DownloadResult(index, url, False, None, f"HTTP {r.status_code}")

        # Basic check to ensure we are downloading an image
        ct = (r.headers.get("Content-Type") or "").lower()
        if not ct.startswith("image/"):
            return DownloadResult(index, url, False, None, f"Not an image Content-Type: {ct}")

        ext = ext_from_response(url, r)
        filename = f"img_{index:04d}{ext}"  # Use padding for better sorting
        out_path = out_dir / filename

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return DownloadResult(index, url, True, str(out_path), None)
    except Exception as e:
        return DownloadResult(index, url, False, None, str(e))


def main():
    """Main script execution."""
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_out_dir = Path(f"data/mac/macbook_download_{run_timestamp}")
    ensure_dir(parent_out_dir)
    print(f"Starting new run. All images will be saved in: {parent_out_dir}\n")

    for query in TARGET_QUERIES:
        print(f"--- Processing query: \"{query}\" ---")

        query_slug = slugify(query)
        out_dir = parent_out_dir / query_slug
        ensure_dir(out_dir)

        print(f"Saving to: {out_dir}")
        print(f"Images requested: {IMAGES_PER_QUERY}; Threads: {NUM_THREADS}")

        urls = collect_image_urls(query, IMAGES_PER_QUERY)
        if not urls:
            print("No candidate URLs found. Skipping to next query.\n")
            continue

        print(f"Collected {len(urls)} URLs. Starting download...")

        session = requests.Session()
        results: List[DownloadResult] = []

        with tqdm(total=len(urls), desc=f"Downloading \"{query_slug}\"", unit="img") as pbar:
            with cf.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [
                    executor.submit(download_one, i + 1, url, out_dir, session)
                    for i, url in enumerate(urls)
                ]
                for future in cf.as_completed(futures):
                    res: DownloadResult = future.result()
                    results.append(res)
                    pbar.update(1)

        ok_count = sum(1 for r in results if r.ok)
        fail_count = len(results) - ok_count
        print(f"Query \"{query}\" finished. Success: {ok_count}, Failed: {fail_count}\n")

        if fail_count > 0:
            print("Sample failures for this query:")
            failed_samples = [r for r in results if not r.ok][:5]  # Show up to 5 failures
            for r in failed_samples:
                print(f"  - URL: {r.url[:80]}... | Error: {r.error}")
            print("")


if __name__ == "__main__":
    main()
