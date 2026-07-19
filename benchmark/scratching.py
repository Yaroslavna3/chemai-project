import argparse
import csv
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


DEFAULT_OUTPUT = Path("data") / "molecules_by_latin.csv"


def scrape_molecules_by_id(
    base_url: str,
    start_id: int,
    end_id: int,
    output_path: Path,
    delay_seconds: float,
    timeout_seconds: int,
) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    results = []

    print(f"Starting scrape from ID {start_id} to {end_id}...")
    for molecule_id in tqdm(range(start_id, end_id + 1), desc="Scraping molecules"):
        url = f"{base_url}{molecule_id}"
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            if response.status_code == 404:
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            schema_div = soup.find("div", class_="schema")
            h1_tag = schema_div.find("h1") if schema_div else None
            span_tag = h1_tag.find("span") if h1_tag else None
            if span_tag:
                latin_name = span_tag.get_text(strip=True).strip("()")
                results.append({"id": molecule_id, "latin_name": latin_name})

            time.sleep(delay_seconds)
        except requests.RequestException as exc:
            print(f"\nError fetching ID {molecule_id}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "latin_name"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} molecules to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape molecule Latin names by numeric IDs.")
    parser.add_argument("--base-url", required=True, help="URL prefix before the numeric molecule ID.")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scrape_molecules_by_id(
        base_url=args.base_url,
        start_id=args.start_id,
        end_id=args.end_id,
        output_path=args.output,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
