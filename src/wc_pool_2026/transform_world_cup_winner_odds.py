from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.paths import (
    ResourcePaths,
    build_dated_resource_paths,
    build_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    entrant_teams,
    find_latest_dated_file,
    load_json,
    load_pool_resources,
    normalise_team,
    snapshot_date_stamp,
)


def find_latest_raw_file(paths: ResourcePaths) -> Path:
    return find_latest_dated_file(
        resources=paths,
        subdirectory="raw_api",
        filename="world_cup_winner_odds.json",
    )


def parse_outcomes(events: list[dict]) -> pd.DataFrame:
    rows = []

    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue

                for outcome in market.get("outcomes", []):
                    team = normalise_team(outcome["name"].strip())

                    rows.append(
                        {
                            "team": team,
                            "decimal_odds": float(outcome["price"]),
                            "bookmaker": bookmaker["title"],
                        }
                    )

    if not rows:
        raise RuntimeError("No outright odds found in raw JSON.")

    return pd.DataFrame(rows)


def filter_to_entrant_teams(
    odds_df: pd.DataFrame,
    entrant_teams: set[str],
) -> pd.DataFrame:
    filtered = odds_df[
        odds_df["team"].isin(entrant_teams)
    ].copy()

    missing_teams = sorted(
        entrant_teams - set(filtered["team"])
    )

    if missing_teams:
        raise ValueError(
            f"Missing odds for entrant teams: {missing_teams}"
        )

    return filtered


def build_probabilities(odds_df: pd.DataFrame) -> pd.DataFrame:
    odds_df = odds_df.copy()
    odds_df["implied_prob"] = 1 / odds_df["decimal_odds"]

    team_probs = odds_df.groupby("team", as_index=False).agg(
        raw_implied_prob=("implied_prob", "mean"),
        best_decimal_odds=("decimal_odds", "max"),
        worst_decimal_odds=("decimal_odds", "min"),
        bookmaker_count=("bookmaker", "nunique"),
    )

    team_probs["win_prob"] = (
        team_probs["raw_implied_prob"]
        / team_probs["raw_implied_prob"].sum()
    )

    return (
        team_probs.sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
    )


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
    resources = load_pool_resources(paths.config_dir)
    raw_path = find_latest_raw_file(paths)
    events = load_json(raw_path)

    odds_df = parse_outcomes(events)
    odds_df = filter_to_entrant_teams(
        odds_df=odds_df,
        entrant_teams=entrant_teams(resources.entrants),
    )
    probs_df = build_probabilities(odds_df)

    output = probs_df[
        [
            "team",
            "win_prob",
            "best_decimal_odds",
            "worst_decimal_odds",
            "bookmaker_count",
        ]
    ]
    dated_paths = build_dated_resource_paths(
        resources=paths,
        date_stamp=snapshot_date_stamp(raw_path),
    )

    output_path = (
        dated_paths.input_csv_dir
        / "world_cup_winner_odds.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Read raw JSON from {raw_path}")
    print(f"Wrote CSV to {output_path}")
    print(output.to_string(index=False))
    print("\nProbability sum:", round(output["win_prob"].sum(), 6))
    print("Number of teams:", len(output))


if __name__ == "__main__":
    main()
