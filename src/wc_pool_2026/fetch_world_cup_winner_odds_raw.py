from pathlib import Path
import json
import os

import requests
from dotenv import load_dotenv

from wc_pool_2026.paths import CONFIG_DIR
from wc_pool_2026.common import current_date_stamp

load_dotenv()

API_KEY = os.getenv("API_KEY")

RAW_DIR = CONFIG_DIR / "raw_api"

SPORT_KEY = "soccer_fifa_world_cup_winner"
REGIONS = "uk"
MARKETS = "outrights"
ODDS_FORMAT = "decimal"


def fetch_world_cup_winner_odds() -> list[dict]:
    if not API_KEY:
        raise RuntimeError("Missing api_key in .env")

    response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/",
        params={
            "apiKey": API_KEY,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": ODDS_FORMAT,
        },
        timeout=30,
    )

    response.raise_for_status()

    print(
        "API credits:",
        f"remaining={response.headers.get('x-requests-remaining')}",
        f"used={response.headers.get('x-requests-used')}",
        f"last={response.headers.get('x-requests-last')}",
    )

    return response.json()


def build_output_path() -> Path:
    return RAW_DIR / f"world_cup_winner_odds_{current_date_stamp()}.json"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    events = fetch_world_cup_winner_odds()
    output_path = build_output_path()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    print(f"\nWrote raw JSON to {output_path}")
    print(f"Events returned: {len(events)}")


if __name__ == "__main__":
    main()
