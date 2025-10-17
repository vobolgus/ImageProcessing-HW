#!/usr/bin/env python3
"""
Download the first N images from Google Images for a hardcoded query.

Features:
- Hardcoded query (TARGET), image count (N_IMAGES), and number of threads (NUM_THREADS)
- Saves to /data/$target-$timestamp/img_$n (n starts at 1)
- Multithreaded downloads with basic error handling
- Console progress reporting (tqdm)

Notes:
- This script scrapes Google Images HTML. It may break if Google changes markup
  or if access is rate-limited. Consider using an official API for robustness.
- Ensure you have permissions to write to /data on your system.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import quote_plus

# --------------------------- Configuration ---------------------------
# Hardcoded query and parameters. Adjust as needed.
TARGET = "macbook"  # текстовый запрос (пример)
N_IMAGES = 1000      # сколько картинок скачать
NUM_THREADS = 32    # число потоков (m)

# Timeout settings (seconds)
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

# User-Agent to avoid immediate blocking
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# --------------------------------------------------------------------

@dataclass
class DownloadResult:
    index: int
    url: str
    ok: bool
    path: str | None
    error: str | None


def slugify(value: str) -> str:
    # Simple slugify: lowercase, replace spaces with underscores, remove invalid chars
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]+", "", value)
    return value or "query"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_jpg_url(u: str) -> bool:
    return re.search(r"\.jpe?g(?:$|[?#])", u, flags=re.IGNORECASE) is not None


def is_google_ui_or_thumb(u: str) -> bool:
    # Filter out Google UI assets and thumbnails
    u_low = u.lower()
    return (
        "google.com/images/branding" in u_low
        or "encrypted-tbn" in u_low  # Google thumbnails
        or "gstatic.com/images" in u_low
        or "/favicon" in u_low
    )


def google_image_search_urls(query: str, needed: int, pause: float = 0.8) -> List[str]:
    """Fetch candidate image URLs from Google Images HTML pages.

    Strategy:
      1) Query https://www.google.com/search?tbm=isch for successive pages (ijn=0,1,...)
      2) Parse "data-iurl" and "src" from the results grid
      3) Fallback: regex-scan for image-like URLs (jpg/png/webp)
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    urls: List[str] = []
    seen: Set[str] = set()
    max_pages = 10  # safety

    q = quote_plus(query)
    for ijn in range(max_pages):
        if len(urls) >= needed:
            break
        url = f"https://www.google.com/search?q={q}&tbm=isch&hl=en&ijn={ijn}&tbs=ift:jpg"
        try:
            resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.status_code != 200:
                # Try a short backoff and continue
                time.sleep(pause)
                continue
        except Exception:
            time.sleep(pause)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) Preferred: take URLs from result grid only; ignore page UI images
        for tag in soup.select("div.isv-r img[data-iurl], div.isv-r img[src]"):
            candidate = tag.get("data-iurl") or tag.get("src") or tag.get("data-src")
            if not candidate:
                continue
            if candidate.startswith("data:"):
                continue
            if candidate.startswith("/"):
                continue
            if not candidate.startswith("http"):
                continue
            if is_google_ui_or_thumb(candidate):
                continue
            if not is_jpg_url(candidate):
                continue
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
                if len(urls) >= needed:
                    break
        if len(urls) >= needed:
            break

        # 2) Fallback: scrape script text for JPG urls only
        if len(urls) < needed:
            # Look for .jpg/.jpeg URLs
            img_like = set(re.findall(r"https?://[^\s'\"<>]+\.jpe?g(?:\?[^\s'\"<>]*)?", resp.text, flags=re.IGNORECASE))
            for u in img_like:
                if is_google_ui_or_thumb(u):
                    continue
                if not is_jpg_url(u):
                    continue
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= needed:
                        break

        time.sleep(pause)

    return urls[:needed]


def ext_from_response(url: str, resp: requests.Response) -> str:
    # Try from Content-Type, then from URL
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "image/jpeg" in ct or "/jpg" in ct:
        return ".jpg"
    if "image/png" in ct:
        return ".png"
    if "image/webp" in ct:
        return ".webp"
    if "image/gif" in ct:
        return ".gif"

    m = re.search(r"\.(jpe?g|png|webp|gif)(?:\?|$)", url, flags=re.IGNORECASE)
    if m:
        return "." + m.group(1).lower()
    return ".jpg"


def download_one(index: int, url: str, out_dir: Path, session: requests.Session, lock: threading.Lock) -> DownloadResult:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.google.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        r = session.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True)
        if r.status_code != 200:
            return DownloadResult(index, url, False, None, f"HTTP {r.status_code}")
        # Verify JPEG-only content
        ct = (r.headers.get("Content-Type") or "").lower()
        if ("image/jpeg" not in ct) and (not re.search(r"\.jpe?g(?:$|[?#])", url, flags=re.IGNORECASE)):
            return DownloadResult(index, url, False, None, "non-jpg content")
        filename = f"img_{index}.jpg"
        out_path = out_dir / filename
        # Write to disk
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return DownloadResult(index, url, True, str(out_path), None)
    except Exception as e:
        return DownloadResult(index, url, False, None, str(e))


def main():
    target = TARGET
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"data/{target}-{N_IMAGES}-{timestamp}")
    ensure_dir(out_dir)

    print(f"Target: {target}")
    print(f"Saving to: {out_dir}")
    print(f"Images requested: {N_IMAGES}; Threads: {NUM_THREADS}")

    print("Collecting image URLs from Google Images ...")
    urls = google_image_search_urls(target, N_IMAGES)
    print(f"Collected {len(urls)} candidate URLs")

    # Prepare session and download pool
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: List[DownloadResult] = []
    lock = threading.Lock()

    with tqdm(total=len(urls), desc="Downloading", unit="img") as pbar:
        with cf.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(download_one, i + 1, url, out_dir, session, lock)
                for i, url in enumerate(urls)
            ]
            for future in cf.as_completed(futures):
                res: DownloadResult = future.result()
                results.append(res)
                if res.ok:
                    pbar.update(1)
                else:
                    # Consider failed downloads still move the bar to reflect attempts
                    pbar.update(1)
                    pbar.set_postfix_str("failures present")

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print(f"Done. Success: {ok_count}, Failed: {fail_count}")
    if fail_count:
        print("Sample failures:")
        for r in results:
            if not r.ok:
                print(f"  #{r.index}: {r.url} -> {r.error}")
                # limit printed failures
                fail_count -= 1
                if fail_count <= 0:
                    break


if __name__ == "__main__":
    main()
