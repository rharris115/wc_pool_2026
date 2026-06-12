import json
import os
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

from wc_pool_2026.paths import (
    build_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import current_date_stamp

load_dotenv()

API_KEY = os.getenv("API_KEY")

SPORT_KEY = "soccer_fifa_world_cup"


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
    paths = build_resource_paths(resources_path)

    paths.raw_api_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores/",
        params={
            "apiKey": API_KEY,
            "daysFrom": 3,
            "dateFormat": "iso",
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

    data = response.json()

    output_path = (
        paths.raw_api_dir
        / f"world_cup_scores_{current_date_stamp()}.json"
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(data)} score records to {output_path}")


if __name__ == "__main__":
    main()
