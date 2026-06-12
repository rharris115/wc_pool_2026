from pathlib import Path

import pandas as pd

try:
    from .paths import CONFIG_DIR, CSV_DIR
except ImportError:
    from paths import CONFIG_DIR, CSV_DIR

try:
    from .common import (
        entrant_teams,
        extract_date_stamp,
        find_latest_file,
        load_json,
        normalise_team,
    )
except ImportError:
    from common import (
        entrant_teams,
        extract_date_stamp,
        find_latest_file,
        load_json,
        normalise_team,
    )

RAW_DIR = CONFIG_DIR / "raw_api"


def build_output_path(raw_path: Path) -> Path:
    date_stamp = extract_date_stamp(raw_path)
    return CSV_DIR / f"team_win_probs_{date_stamp}.csv"


def find_latest_raw_file() -> Path:
    return find_latest_file(RAW_DIR, "world_cup_winner_odds_*.json")


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


def main() -> None:
    raw_path = find_latest_raw_file()
    events = load_json(raw_path)

    odds_df = parse_outcomes(events)
    odds_df = filter_to_entrant_teams(
        odds_df=odds_df,
        entrant_teams=entrant_teams(),
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

    output_path = build_output_path(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Read raw JSON from {raw_path}")
    print(f"Wrote CSV to {output_path}")
    print(output.to_string(index=False))
    print("\nProbability sum:", round(output["win_prob"].sum(), 6))
    print("Number of teams:", len(output))


if __name__ == "__main__":
    main()
