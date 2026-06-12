from pathlib import Path

import pandas as pd

try:
    from .paths import CONFIG_DIR, CSV_DIR
except ImportError:
    from paths import CONFIG_DIR, CSV_DIR

try:
    from .common import (
        extract_date_stamp,
        find_latest_file,
        load_json,
        normalise_team,
    )
except ImportError:
    from common import (
        extract_date_stamp,
        find_latest_file,
        load_json,
        normalise_team,
    )

RAW_DIR = CONFIG_DIR / "raw_api"


def build_output_path(raw_path: Path) -> Path:
    date_stamp = extract_date_stamp(raw_path)
    return CSV_DIR / f"group_match_outcome_probs_{date_stamp}.csv"


def find_latest_raw_file() -> Path:
    return find_latest_file(RAW_DIR, "world_cup_match_odds_*.json")


def normalise_probs(prices: dict[str, float]) -> dict[str, float]:
    raw = {
        outcome: 1 / price
        for outcome, price in prices.items()
    }

    total = sum(raw.values())

    return {
        outcome: prob / total
        for outcome, prob in raw.items()
    }


def extract_bookmaker_h2h_probs(event: dict) -> list[dict[str, float]]:
    bookmaker_probs = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue

            prices = {
                normalise_team(outcome["name"]): float(outcome["price"])
                for outcome in market["outcomes"]
            }

            bookmaker_probs.append(normalise_probs(prices))

    return bookmaker_probs


def consensus_h2h_probs(event: dict) -> dict[str, float]:
    team_1 = normalise_team(event["home_team"])
    team_2 = normalise_team(event["away_team"])

    bookmaker_probs = extract_bookmaker_h2h_probs(event)

    if not bookmaker_probs:
        raise RuntimeError(
            f"No h2h markets found for {team_1} vs {team_2}"
        )

    expected_outcomes = {team_1, team_2, "Draw"}

    valid_probs = [
        probs
        for probs in bookmaker_probs
        if expected_outcomes <= set(probs)
    ]

    if not valid_probs:
        raise RuntimeError(
            f"No complete h2h markets found for {team_1} vs {team_2}"
        )

    return {
        outcome: sum(probs[outcome] for probs in valid_probs) / len(valid_probs)
        for outcome in expected_outcomes
    }


def parse_match_outcomes(events: list[dict]) -> pd.DataFrame:
    rows = []

    for match_number, event in enumerate(events, start=1):
        team_1 = normalise_team(event["home_team"])
        team_2 = normalise_team(event["away_team"])

        probs = consensus_h2h_probs(event)

        p_team_1_win = probs[team_1]
        p_draw = probs["Draw"]
        p_team_2_win = probs[team_2]

        bookmaker_count = len(extract_bookmaker_h2h_probs(event))

        rows.append(
            {
                "match_id": event["id"],
                "match_number": match_number,
                "commence_time": event["commence_time"],
                "team": team_1,
                "opponent": team_2,
                "p_win": p_team_1_win,
                "p_draw": p_draw,
                "p_loss": p_team_2_win,
                "bookmaker_count": bookmaker_count,
            }
        )

        rows.append(
            {
                "match_id": event["id"],
                "match_number": match_number,
                "commence_time": event["commence_time"],
                "team": team_2,
                "opponent": team_1,
                "p_win": p_team_2_win,
                "p_draw": p_draw,
                "p_loss": p_team_1_win,
                "bookmaker_count": bookmaker_count,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["commence_time", "match_number", "team"])
        .reset_index(drop=True)
    )


def main() -> None:
    raw_path = find_latest_raw_file()
    events = load_json(raw_path)

    output = parse_match_outcomes(events)

    output_path = build_output_path(raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_path, index=False)

    print(f"Read raw JSON from {raw_path}")
    print(f"Wrote CSV to {output_path}")
    print(output.to_string(index=False))
    print("\nNumber of fixtures:", output["match_id"].nunique())
    print("Number of team-match rows:", len(output))


if __name__ == "__main__":
    main()
