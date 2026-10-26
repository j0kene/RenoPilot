"""Download the HDB renovation-rule sources behind the compliance knowledge base (dated snapshot).

    uv run python data/scripts/fetch_hdb_sources.py

Each source is saved as raw HTML and as plain text (data/raw/hdb/<id>.html / .txt), and every
attempt, failed or not, is recorded in data/compliance/sources.json with its status and date.
hdb.gov.sg renders its guideline text with JavaScript (the HTML only has headings),
so the text is decoded from the script data (embedded_html); the rule cards
(data/compliance/hdb_rules.jsonl) quote it, checked by tests/test_compliance_kb.py.

Polite by construction: one request at a time with a pause, a dozen pages in total.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from renopilot import config  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RenoPilot-CS621-student-project"}
PAUSE_S = 1.5
RAW = config.DATA / "raw" / "hdb"
MANIFEST = config.DATA / "compliance" / "sources.json"

# id -> (url, official). official: HDB or another government site; the rest are secondary guides.
SOURCES = {
    "hdb_building_works": ("https://www.hdb.gov.sg/managing-my-home/renovation-and-maintenance/renovation/"
                           "renovation-guidelines/building-works", True),
    "hdb_important_info": ("https://www.hdb.gov.sg/managing-my-home/renovation-and-maintenance/renovation/"
                           "important-information-on-renovations", True),
    "hdb_permit": ("https://www.hdb.gov.sg/managing-my-home/renovation-and-maintenance/renovation/"
                   "application-for-a-renovation-permit", True),
    "mynicehome_reno": ("https://www.mynicehome.gov.sg/get-settled-in/hdb-renovation-guide/", True),
    "homeanddecor_dos": ("https://www.homeanddecor.com.sg/how-to/home-tips-the-dos-and-donts-when-renovating-your-hdb-flat", False),
    "hydroworks_3yr": ("https://hydroworks.com.sg/blog/hdb-renovation-waterproofing-rules/", False),
    "leongyik_wet_area": ("https://www.leongyik.com.sg/blog/hdb-wet-area-floor-tiling-guidelines/", False),
    "homeexpo_overlay": ("https://homeexpo.asia/vinyl-flooring/hdb-flooring-overlay-regulations-your-complete-guide-2026/", False),
    "shiokliving_tiles": ("https://www.shiokliving.com/home-improvement/renovation-materials/hdb-approved-tiles/", False),
    "renotalk_shelter": ("https://www.renotalk.com/article/guide-to-the-standard-hdb-household-shelter", False),
    "aura_hacking": ("https://auradesignandco.com/hdb-hacking-rules-what-you-can-and-cannot-hack/", False),
    "elumi_permit": ("https://www.elumihome.com/blog/hdb-renovation-permit-guide", False),
}


class _Text(HTMLParser):
    """Visible text, one block element per line; scripts, styles and page chrome dropped."""
    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "tr", "div", "section", "article", "br", "td"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4") and not self.skip:
            self.out.append("## ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    lines = (re.sub(r"\s+", " ", ln).strip() for ln in "".join(p.out).splitlines())
    return "\n".join(ln for ln in lines if ln and ln != "##")


def embedded_html(html: str) -> str:
    """hdb.gov.sg ships its accordion text (the actual rules) as JSON-escaped HTML inside script
    data, not as page markup: decode every JSON string that holds markup."""
    out = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*\\u003c(?:[^"\\]|\\.)*)"', html):
        try:
            out.append(json.loads(f'"{m.group(1)}"'))
        except json.JSONDecodeError:
            continue
    return "\n".join(out)


def page_text(html: str) -> str:
    """Visible text plus embedded rich text, each line once, in page order."""
    seen, lines = set(), []
    for ln in (html_to_text(html) + "\n" + html_to_text(embedded_html(html))).splitlines():
        if ln not in seen:
            seen.add(ln)
            lines.append(ln)
    return "\n".join(lines)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    manifest = {}
    for sid, (url, official) in SOURCES.items():
        entry = {"url": url, "official": official, "date_checked": today}
        try:
            r = requests.get(url, headers=UA, timeout=30)
            entry["status"] = r.status_code
            if r.ok and "html" in r.headers.get("content-type", ""):
                text = page_text(r.text)
                (RAW / f"{sid}.html").write_text(r.text, encoding="utf-8")
                (RAW / f"{sid}.txt").write_text(f"# {sid}\n# {url}\n\n{text}\n", encoding="utf-8")
                entry |= {"chars": len(text), "sha256": hashlib.sha256(r.content).hexdigest()[:16]}
        except requests.RequestException as e:
            entry["status"] = f"error: {type(e).__name__}"
        manifest[sid] = entry
        print(f"{sid:20} {entry['status']} {entry.get('chars', '')}")
        time.sleep(PAUSE_S)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
