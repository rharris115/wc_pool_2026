from pathlib import Path
import json
import os

import click
import requests
from dotenv import load_dotenv

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import current_date_stamp

load_dotenv()

API_KEY = os.getenv("API_KEY")

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


@click.command()
@click.argument(
    "resources_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    default=default_resources_path(),
    required=False,
)
def main(resources_path: Path) -> None:
    resources_path = resources_path.expanduser().resolve()
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=current_date_stamp(),
    )
    dated_paths.raw_api_dir.mkdir(parents=True, exist_ok=True)

    events = fetch_world_cup_winner_odds()
    output_path = dated_paths.raw_api_dir / "world_cup_winner_odds.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    print(f"\nWrote raw JSON to {output_path}")
    print(f"Events returned: {len(events)}")


if __name__ == "__main__":
    main()
