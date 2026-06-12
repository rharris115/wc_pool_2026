from pathlib import Path
import math

import click
import pandas as pd

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    find_latest_dated_file,
    load_json,
    normalise_team,
    snapshot_date_stamp,
    sort_match_rows,
)

TOTAL_GOALS_MARKET_POINT = 2.5
MIN_EXPECTED_GOALS = 0.05
MAX_EXPECTED_GOALS = 8.0
XG_GRID_SIZE = 1000


def find_latest_raw_file(resources_path: Path) -> Path:
    return find_latest_dated_file(
        resources_path=resources_path,
        subdirectory="raw_api",
        filename="world_cup_match_odds.json",
    )


def normalise_probs(prices: dict[str, float]) -> dict[str, float]:
    raw = {outcome: 1 / price for outcome, price in prices.items()}

    total = sum(raw.values())

    return {outcome: prob / total for outcome, prob in raw.items()}


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
        raise RuntimeError(f"No h2h markets found for {team_1} vs {team_2}")

    expected_outcomes = {team_1, team_2, "Draw"}

    valid_probs = [
        probs for probs in bookmaker_probs if expected_outcomes <= set(probs)
    ]

    if not valid_probs:
        raise RuntimeError(f"No complete h2h markets found for {team_1} vs {team_2}")

    return {
        outcome: sum(probs[outcome] for probs in valid_probs) / len(valid_probs)
        for outcome in expected_outcomes
    }


def extract_bookmaker_total_goal_probs(event: dict) -> list[dict[str, float]]:
    bookmaker_probs = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue

            prices = {
                outcome["name"]: float(outcome["price"])
                for outcome in market["outcomes"]
                if outcome.get("point") == TOTAL_GOALS_MARKET_POINT
            }

            if {"Over", "Under"} <= set(prices):
                bookmaker_probs.append(normalise_probs(prices))

    return bookmaker_probs


def consensus_over_25_prob(event: dict) -> float:
    bookmaker_probs = extract_bookmaker_total_goal_probs(event)

    if not bookmaker_probs:
        raise RuntimeError(
            f"No totals market found for {event['home_team']} vs {event['away_team']}"
        )

    return sum(probs["Over"] for probs in bookmaker_probs) / len(bookmaker_probs)


def poisson_under_25_prob(expected_goals: float) -> float:
    return math.exp(-expected_goals) * (1 + expected_goals + (expected_goals**2 / 2))


def infer_expected_total_goals(p_over_25: float) -> float:
    low = MIN_EXPECTED_GOALS
    high = MAX_EXPECTED_GOALS

    for _ in range(60):
        mid = (low + high) / 2
        mid_over = 1 - poisson_under_25_prob(mid)

        if mid_over < p_over_25:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def poisson_probs(expected_goals: float, max_goals: int) -> list[float]:
    probs = [math.exp(-expected_goals)]

    for goals in range(1, max_goals + 1):
        probs.append(probs[-1] * expected_goals / goals)

    return probs


def h2h_probs_from_xg(
    team_xg: float,
    opponent_xg: float,
) -> dict[str, float]:
    max_goals = max(
        12,
        math.ceil(team_xg + opponent_xg + 8),
    )

    team_goal_probs = poisson_probs(team_xg, max_goals)
    opponent_goal_probs = poisson_probs(opponent_xg, max_goals)

    p_win = 0.0
    p_draw = 0.0
    p_loss = 0.0

    for team_goals, team_prob in enumerate(team_goal_probs):
        for opponent_goals, opponent_prob in enumerate(opponent_goal_probs):
            probability = team_prob * opponent_prob

            if team_goals > opponent_goals:
                p_win += probability
            elif team_goals == opponent_goals:
                p_draw += probability
            else:
                p_loss += probability

    total = p_win + p_draw + p_loss

    return {
        "win": p_win / total,
        "draw": p_draw / total,
        "loss": p_loss / total,
    }


def infer_xg(
    p_team_win: float,
    p_draw: float,
    p_team_loss: float,
    expected_total_goals: float,
) -> tuple[float, float]:
    best_team_xg = expected_total_goals / 2
    best_error = float("inf")

    for i in range(1, XG_GRID_SIZE):
        team_share = i / XG_GRID_SIZE
        team_xg = expected_total_goals * team_share
        opponent_xg = expected_total_goals - team_xg

        probs = h2h_probs_from_xg(
            team_xg=team_xg,
            opponent_xg=opponent_xg,
        )

        error = (
            (probs["win"] - p_team_win) ** 2
            + (probs["draw"] - p_draw) ** 2
            + (probs["loss"] - p_team_loss) ** 2
        )

        if error < best_error:
            best_error = error
            best_team_xg = team_xg

    return best_team_xg, expected_total_goals - best_team_xg


def parse_match_outcomes(events: list[dict]) -> pd.DataFrame:
    rows = []

    for event in events:
        team_1 = normalise_team(event["home_team"])
        team_2 = normalise_team(event["away_team"])

        probs = consensus_h2h_probs(event)

        p_team_1_win = probs[team_1]
        p_draw = probs["Draw"]
        p_team_2_win = probs[team_2]

        expected_total_goals = infer_expected_total_goals(consensus_over_25_prob(event))
        team_1_xg, team_2_xg = infer_xg(
            p_team_win=p_team_1_win,
            p_draw=p_draw,
            p_team_loss=p_team_2_win,
            expected_total_goals=expected_total_goals,
        )

        bookmaker_count = len(extract_bookmaker_h2h_probs(event))

        rows.append(
            {
                "match_id": event["id"],
                "commence_time": event["commence_time"],
                "team": team_1,
                "opponent": team_2,
                "p_win": p_team_1_win,
                "p_draw": p_draw,
                "p_loss": p_team_2_win,
                "team_xg": team_1_xg,
                "opponent_xg": team_2_xg,
                "expected_total_goals": expected_total_goals,
                "bookmaker_count": bookmaker_count,
            }
        )

        rows.append(
            {
                "match_id": event["id"],
                "commence_time": event["commence_time"],
                "team": team_2,
                "opponent": team_1,
                "p_win": p_team_2_win,
                "p_draw": p_draw,
                "p_loss": p_team_1_win,
                "team_xg": team_2_xg,
                "opponent_xg": team_1_xg,
                "expected_total_goals": expected_total_goals,
                "bookmaker_count": bookmaker_count,
            }
        )

    return sort_match_rows(pd.DataFrame(rows))


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
    raw_path = find_latest_raw_file(resources_path)
    events = load_json(raw_path)

    output = parse_match_outcomes(events)
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp(raw_path),
    )

    output_path = dated_paths.input_csv_dir / "group_match_outcome_probs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_path, index=False)

    print(f"Read raw JSON from {raw_path}")
    print(f"Wrote CSV to {output_path}")
    print(output.to_string(index=False))
    print("\nNumber of fixtures:", output["match_id"].nunique())
    print("Number of team-match rows:", len(output))


if __name__ == "__main__":
    main()
