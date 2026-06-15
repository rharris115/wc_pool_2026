from pathlib import Path
import math

import click
import pandas as pd
from scipy.optimize import minimize

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    find_dated_file,
    find_latest_dated_file,
    load_json,
    normalise_team,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    sort_match_rows,
)

TOTAL_GOALS_MARKET_POINT = 2.5
FALLBACK_P_OVER_25 = 0.45
TOTALS_WEIGHT = 1.0
MAX_GOALS_FOR_FIT = 12
EPS = 1e-12


def find_latest_raw_file(resources_path: Path) -> Path:
    return find_latest_dated_file(
        resources_path=resources_path,
        subdirectory="raw_api",
        filename="world_cup_match_odds.json",
    )


def find_raw_file(
    resources_path: Path,
    date_stamp: str | None = None,
) -> Path:
    return find_dated_file(
        resources_path=resources_path,
        subdirectory="raw_api",
        filename="world_cup_match_odds.json",
        date_stamp=date_stamp,
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


def consensus_total_goal_probs(event: dict) -> tuple[float, float, int]:
    bookmaker_probs = extract_bookmaker_total_goal_probs(event)

    if not bookmaker_probs:
        return FALLBACK_P_OVER_25, 1 - FALLBACK_P_OVER_25, 0

    p_over_25 = sum(probs["Over"] for probs in bookmaker_probs) / len(bookmaker_probs)

    return p_over_25, 1 - p_over_25, len(bookmaker_probs)


def poisson_under_25_prob(expected_goals: float) -> float:
    return math.exp(-expected_goals) * (1 + expected_goals + (expected_goals**2 / 2))


def infer_expected_total_goals(p_over_25: float) -> float:
    low = 0.0
    high = 1.0

    while 1 - poisson_under_25_prob(high) < p_over_25:
        high *= 2

    for _ in range(60):
        mid = (low + high) / 2
        mid_over = 1 - poisson_under_25_prob(mid)

        if mid_over < p_over_25:
            low = mid
        else:
            high = mid

    return max((low + high) / 2, EPS)


def poisson_probs(expected_goals: float, max_goals: int) -> list[float]:
    probs = [math.exp(-expected_goals)]

    for goals in range(1, max_goals + 1):
        probs.append(probs[-1] * expected_goals / goals)

    return probs


def model_probs_from_xg(lambda_1: float, lambda_2: float) -> dict[str, float]:
    team_goal_probs = poisson_probs(lambda_1, MAX_GOALS_FOR_FIT)
    opponent_goal_probs = poisson_probs(lambda_2, MAX_GOALS_FOR_FIT)

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

    h2h_total = p_win + p_draw + p_loss
    expected_total_goals = lambda_1 + lambda_2
    p_under_25 = poisson_under_25_prob(expected_total_goals)

    return {
        "win": p_win / h2h_total,
        "draw": p_draw / h2h_total,
        "loss": p_loss / h2h_total,
        "under_25": p_under_25,
        "over_25": 1 - p_under_25,
    }


def fit_market_implied_xg(
    p_team_1_win: float,
    p_draw: float,
    p_team_2_win: float,
    p_over_25: float,
    p_under_25: float,
) -> dict[str, float]:
    initial_total_goals = infer_expected_total_goals(p_over_25)
    team_1_share_denominator = p_team_1_win + p_team_2_win
    team_1_share = (
        p_team_1_win / team_1_share_denominator if team_1_share_denominator > 0 else 0.5
    )
    lambda_1_initial = max(initial_total_goals * team_1_share, EPS)
    lambda_2_initial = max(initial_total_goals * (1 - team_1_share), EPS)

    def negative_log_likelihood(theta) -> float:
        lambda_1 = math.exp(theta[0])
        lambda_2 = math.exp(theta[1])
        model_probs = model_probs_from_xg(lambda_1=lambda_1, lambda_2=lambda_2)

        return -(
            p_team_1_win * math.log(model_probs["win"] + EPS)
            + p_draw * math.log(model_probs["draw"] + EPS)
            + p_team_2_win * math.log(model_probs["loss"] + EPS)
            + TOTALS_WEIGHT
            * (
                p_over_25 * math.log(model_probs["over_25"] + EPS)
                + p_under_25 * math.log(model_probs["under_25"] + EPS)
            )
        )

    result = minimize(
        negative_log_likelihood,
        x0=[math.log(lambda_1_initial), math.log(lambda_2_initial)],
        method="L-BFGS-B",
    )

    lambda_1 = math.exp(result.x[0])
    lambda_2 = math.exp(result.x[1])
    fitted_probs = model_probs_from_xg(lambda_1=lambda_1, lambda_2=lambda_2)

    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "expected_total_goals": lambda_1 + lambda_2,
        "fit_model_p_win": fitted_probs["win"],
        "fit_model_p_draw": fitted_probs["draw"],
        "fit_model_p_loss": fitted_probs["loss"],
        "fit_model_p_over_25": fitted_probs["over_25"],
        "fit_loss": float(result.fun),
    }


def parse_match_outcomes(events: list[dict]) -> pd.DataFrame:
    rows = []

    for event in events:
        team_1 = normalise_team(event["home_team"])
        team_2 = normalise_team(event["away_team"])

        probs = consensus_h2h_probs(event)

        p_team_1_win = probs[team_1]
        p_draw = probs["Draw"]
        p_team_2_win = probs[team_2]

        p_over_25, p_under_25, totals_bookmaker_count = consensus_total_goal_probs(
            event
        )
        fit = fit_market_implied_xg(
            p_team_1_win=p_team_1_win,
            p_draw=p_draw,
            p_team_2_win=p_team_2_win,
            p_over_25=p_over_25,
            p_under_25=p_under_25,
        )
        team_1_xg = fit["lambda_1"]
        team_2_xg = fit["lambda_2"]

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
                "expected_total_goals": fit["expected_total_goals"],
                "p_over_25": p_over_25,
                "p_under_25": p_under_25,
                "totals_bookmaker_count": totals_bookmaker_count,
                "fit_model_p_win": fit["fit_model_p_win"],
                "fit_model_p_draw": fit["fit_model_p_draw"],
                "fit_model_p_loss": fit["fit_model_p_loss"],
                "fit_model_p_over_25": fit["fit_model_p_over_25"],
                "fit_loss": fit["fit_loss"],
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
                "expected_total_goals": fit["expected_total_goals"],
                "p_over_25": p_over_25,
                "p_under_25": p_under_25,
                "totals_bookmaker_count": totals_bookmaker_count,
                "fit_model_p_win": fit["fit_model_p_loss"],
                "fit_model_p_draw": fit["fit_model_p_draw"],
                "fit_model_p_loss": fit["fit_model_p_win"],
                "fit_model_p_over_25": fit["fit_model_p_over_25"],
                "fit_loss": fit["fit_loss"],
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
@click.option(
    "--snapshot-date-stamp",
    "--date-stamp",
    "snapshot_date_stamp",
    default=None,
    help=(
        "Dated resource snapshot to read/write, for example 20260614. "
        "Defaults to inferring the latest matching snapshot from resources."
    ),
)
def main(resources_path: Path, snapshot_date_stamp: str | None) -> None:
    resources_path = resources_path.expanduser().resolve()
    raw_path = find_raw_file(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp,
    )
    events = load_json(raw_path)

    output = parse_match_outcomes(events)
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(raw_path),
    )

    output_path = dated_paths.input_csv_dir / "group_match_outcome_probs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_path, index=False)

    text_output = "\n".join(
        [
            f"Read raw JSON from {raw_path}",
            f"Wrote CSV to {output_path}",
            output.to_string(index=False),
            "",
            f"Number of fixtures: {output['match_id'].nunique()}",
            f"Number of team-match rows: {len(output)}",
        ]
    )
    click.echo(text_output)


if __name__ == "__main__":
    main()
