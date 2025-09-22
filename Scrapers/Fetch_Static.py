# scrapers/fetch_static.py
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

URL = "https://api.freshservice.com/#ticket_attributes"
OUT_FILE = "data/attributes.jsonl"

def scrape_table(url=URL):
    resp = requests.get(url, headers={"User-Agent":"fresh-scraper/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Locate the table that contains ticket attributes. Adjust selector if needed.
    # Example: find by caption or by the anchor section
    anchor = soup.find(id="ticket_attributes")
    table = anchor.find_next("table") if anchor else soup.find("table")
    if not table:
        raise SystemExit("Could not find attributes table")

    rows = table.find_all("tr")
    header_cells = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th","td"])]
    # expected headers: attribute, type, description, supported product(s)
    out = []
    for i, tr in enumerate(rows[1:], start=1):
        tds = tr.find_all(["td","th"])
        if len(tds) < 3:
            continue        
        # ensure mapping by position if headers vary
        attribute = tds[0].get_text(" ", strip=True)
        typ = tds[1].get_text(" ", strip=True)
        description = tds[2].get_text(" ", strip=True)
        support = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
        doc = {
            "attribute": attribute,
            "type": typ,
            "description": description,
            "supported_products": support,
            "source_url": url,
            "anchor": "ticket_attributes",
            "position": i
        }
        out.append(doc)

    # persist as JSONL
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for doc in out:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} docs to {OUT_FILE}")
    return out

if __name__ == "__main__":
    scrape_table()
