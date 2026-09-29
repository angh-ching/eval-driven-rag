"""Download football-analytics docs into corpus/.

Re-runnable: same input, same output.
"""
import json
import re

import requests
from bs4 import BeautifulSoup

BASE = "https://socceraction.readthedocs.io/en/latest/documentation/"

SOURCES = [
    "intro.html",
    "install.html",
    "faq.html",
    "data/index.html",
    "data/statsbomb.html",
    "data/wyscout.html",
    "data/opta.html",
    "spadl/index.html",
    "spadl/spadl.html",
    "spadl/atomic_spadl.html",
    "valuing_actions/index.html",
    "valuing_actions/xT.html",
    "valuing_actions/vaep.html",
    "valuing_actions/atomic_vaep.html",
]

# Template furniture, removed by class where we know it.
STRIP = [
    ".bottom-of-page",
    ".related-pages",
    ".header-left",
    ".header-right",
    ".sidebar-container",
    ".view-this-page",
]


def clean(soup):
    """Extract article text, dropping chrome and preserving code blocks."""
    for sel in STRIP:
        for el in soup.select(sel):
            el.decompose()

    # Flatten code blocks BEFORE get_text, or tags shred them one token
    # per line.
    for pre in soup.select("div.highlight, pre"):
        pre.replace_with("\nCODE:\n" + pre.get_text() + "\n")

    text = soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def drop_leading_chrome(text, title):
    """Theme nav sits above the page title. Cut everything before it.

    Position-based, not class-based, so a theme change doesn't break it.
    """
    return text.lstrip()


def main():
    manifest = []
    for path in SOURCES:
        url = BASE + path
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        article = soup.select_one(".article-container article") or soup.select_one("article")
        h1 = (article.select_one("h1")
              or article.select_one("section > p")
              or article.select_one("section")) if article else None
        title = re.sub(r"¶|\s+", " ", h1.get_text(strip=True)).strip() if h1 else ""
        text = drop_leading_chrome(clean(article or soup), title)

        doc_id = path.replace("/", "-").replace(".html", "")
        with open(f"corpus/{doc_id}.txt", "w") as f:
            f.write(text)

        manifest.append({"id": doc_id, "url": url, "title": title,
                         "chars": len(text)})
        print(f"{doc_id:28} {len(text):>6} chars  {title[:30]}")

    with open("corpus/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
