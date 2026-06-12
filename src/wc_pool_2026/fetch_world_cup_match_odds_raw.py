from datetime import datetime
from pathlib import Path
import json
import os

import requests
from dotenv import load_dotenv

try:
    from .paths import CONFIG_DIR, ENV_FILE
except ImportError:
    from paths import CONFIG_DIR, ENV_FILE

load_dotenv(ENV_FILE)

API_KEY = os.getenv("api_key")

OUTPUT_DIR = CONFIG_DIR / "raw_api"

SPORT_KEY = "soccer_fifa_world_cup"
REGIONS = "uk"
MARKETS = "h2h,totals"
ODDS_FORMAT = "decimal"


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Missing api_key in .env"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    events = response.json()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"world_cup_match_odds_{timestamp}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            events,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nWrote {len(events)} events to:"
    )
    print(output_path)

    if events:
        print(
            "\nFirst fixture:"
        )
        print(
            f"{events[0].get('home_team')} "
            f"vs "
            f"{events[0].get('away_team')}"
        )


if __name__ == "__main__":
    main()
