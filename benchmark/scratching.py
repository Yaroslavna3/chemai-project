import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm.auto import tqdm
from pathlib import Path

DATAPATH = Path.cwd() / 'data'
OUTPUT_FILE = 'molecules_by_id.csv'


def scrape_molecules_by_id(start_id, end_id):
    base_url = "https://www.vidal.ru/drugs/molecule/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    results = []

    print(f"Starting scrape from ID {start_id} to {end_id}...")

    # Using tqdm for progress tracking
    for i in tqdm(range(start_id, end_id + 1), desc="Scraping molecules"):
        url = f"{base_url}{i}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                schema_div = soup.find('div', class_='schema')
                if schema_div:
                    h1_tag = schema_div.find('h1')
                    if h1_tag:
                        span_tag = h1_tag.find('span')
                        if span_tag:
                            latin_name = span_tag.get_text(strip=True).strip('()')
                            results.append({'id': i, 'latin_name': latin_name})
            elif response.status_code == 404:
                continue

            time.sleep(0.2) # Slightly reduced delay since it's a large range

        except Exception as e:
            print(f"\nError fetching ID {i}: {e}")

    # Save results to CSV
    with open(DATAPATH / OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'latin_name'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} molecules to {DATAPATH / OUTPUT_FILE}")

# Scrape the full range
scrape_molecules_by_id(1, 4000)