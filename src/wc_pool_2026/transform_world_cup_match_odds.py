from pathlib import Path
import math
from typing import TypedDict

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
    outcome_probs_from_xg,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    sort_match_rows,
)

FALLBACK_EXPECTED_TOTAL_GOALS = 2.5
TOTALS_WEIGHT = 1.0
EPS = 1e-12


class BookmakerTotalGoalProb(TypedDict):
    point: float
    Over: float
    Under: float


class ConsensusTotalGoalProb(BookmakerTotalGoalProb):
    bookmaker_count: int


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


def extract_bookmaker_total_goal_probs(event: dict) -> list[BookmakerTotalGoalProb]:
    bookmaker_probs = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue

            points = sorted(
                {
                    float(outcome["point"])
                    for outcome in market["outcomes"]
                    if "point" in outcome
                }
            )

            for point in points:
                prices = {
                    outcome["name"]: float(outcome["price"])
                    for outcome in market["outcomes"]
                    if outcome.get("point") == point
                }

                if {"Over", "Under"} <= set(prices):
                    probs = normalise_probs(prices)
                    bookmaker_probs.append(
                        {
                            "point": point,
                            "Over": probs["Over"],
                            "Under": probs["Under"],
                        }
                    )

    return bookmaker_probs


def consensus_total_goal_probs(event: dict) -> list[ConsensusTotalGoalProb]:
    bookmaker_probs = extract_bookmaker_total_goal_probs(event)

    if not bookmaker_probs:
        return []

    points = sorted({probs["point"] for probs in bookmaker_probs})
    consensus_probs = []

    for point in points:
        point_probs = [probs for probs in bookmaker_probs if probs["point"] == point]
        p_over = sum(probs["Over"] for probs in point_probs) / len(point_probs)
        consensus_probs.append(
            {
                "point": point,
                "Over": p_over,
                "Under": 1 - p_over,
                "bookmaker_count": len(point_probs),
            }
        )

    return consensus_probs


def poisson_under_prob(expected_goals: float, point: float) -> float:
    max_under_goals = math.floor(point)
    probability = math.exp(-expected_goals)
    total_probability = probability

    for goals in range(1, max_under_goals + 1):
        probability *= expected_goals / goals
        total_probability += probability

    return total_probability


def infer_expected_total_goals(p_over: float, point: float) -> float:
    low = 0.0
    high = 1.0

    while 1 - poisson_under_prob(high, point) < p_over:
        high *= 2

    for _ in range(60):
        mid = (low + high) / 2
        mid_over = 1 - poisson_under_prob(mid, point)

        if mid_over < p_over:
            low = mid
        else:
            high = mid

    return max((low + high) / 2, EPS)


def model_total_goal_probs(expected_total_goals: float, point: float) -> dict[str, float]:
    p_under = poisson_under_prob(
        expected_goals=expected_total_goals,
        point=point,
    )

    return {
        "Under": p_under,
        "Over": 1 - p_under,
    }


def fit_market_implied_xg(
    p_team_1_win: float,
    p_draw: float,
    p_team_2_win: float,
    total_goal_probs: list[ConsensusTotalGoalProb],
) -> dict[str, float]:
    if total_goal_probs:
        initial_total_goals = sum(
            infer_expected_total_goals(
                p_over=probs["Over"],
                point=probs["point"],
            )
            for probs in total_goal_probs
        ) / len(total_goal_probs)
    else:
        initial_total_goals = FALLBACK_EXPECTED_TOTAL_GOALS
    team_1_share_denominator = p_team_1_win + p_team_2_win
    team_1_share = (
        p_team_1_win / team_1_share_denominator if team_1_share_denominator > 0 else 0.5
    )
    lambda_1_initial = max(initial_total_goals * team_1_share, EPS)
    lambda_2_initial = max(initial_total_goals * (1 - team_1_share), EPS)

    def negative_log_likelihood(theta) -> float:
        lambda_1 = math.exp(theta[0])
        lambda_2 = math.exp(theta[1])
        model_probs = outcome_probs_from_xg(
            team_xg=lambda_1,
            opponent_xg=lambda_2,
        )
        expected_total_goals = lambda_1 + lambda_2
        totals_log_likelihood = 0.0

        for probs in total_goal_probs:
            model_totals_probs = model_total_goal_probs(
                expected_total_goals=expected_total_goals,
                point=probs["point"],
            )
            totals_log_likelihood += (
                probs["Over"] * math.log(model_totals_probs["Over"] + EPS)
                + probs["Under"] * math.log(model_totals_probs["Under"] + EPS)
            )

        return -(
            p_team_1_win * math.log(model_probs["win"] + EPS)
            + p_draw * math.log(model_probs["draw"] + EPS)
            + p_team_2_win * math.log(model_probs["loss"] + EPS)
            + TOTALS_WEIGHT * totals_log_likelihood
        )

    result = minimize(
        negative_log_likelihood,
        x0=[math.log(lambda_1_initial), math.log(lambda_2_initial)],
        method="L-BFGS-B",
    )

    lambda_1 = math.exp(result.x[0])
    lambda_2 = math.exp(result.x[1])

    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "expected_total_goals": lambda_1 + lambda_2,
        "fit_loss": float(result.fun),
    }


def total_goal_reference_points(
    total_goal_probs: list[ConsensusTotalGoalProb],
) -> str:
    return "|".join(
        f"{probs['point']:g}:{int(probs['bookmaker_count'])}"
        for probs in total_goal_probs
    )


def total_goal_line_count(total_goal_probs: list[ConsensusTotalGoalProb]) -> int:
    return sum(probs["bookmaker_count"] for probs in total_goal_probs)


def total_goal_point_column_label(point: float) -> str:
    return f"{point:g}".replace(".", "_")


def xg_rows(
    event: dict,
    team_1: str,
    team_2: str,
    team_1_xg: float,
    team_2_xg: float,
) -> list[dict]:
    return [
        {
            "match_id": event["id"],
            "commence_time": event["commence_time"],
            "team": team_1,
            "opponent": team_2,
            "team_xg": team_1_xg,
            "opponent_xg": team_2_xg,
        },
        {
            "match_id": event["id"],
            "commence_time": event["commence_time"],
            "team": team_2,
            "opponent": team_1,
            "team_xg": team_2_xg,
            "opponent_xg": team_1_xg,
        },
    ]


def validation_row(
    event: dict,
    team_1: str,
    team_2: str,
    bookmaker_probs: dict[str, float],
    total_goal_probs: list[ConsensusTotalGoalProb],
    fit: dict[str, float],
    h2h_bookmaker_count: int,
) -> dict:
    team_1_xg = fit["lambda_1"]
    team_2_xg = fit["lambda_2"]
    model_outcomes = outcome_probs_from_xg(
        team_xg=team_1_xg,
        opponent_xg=team_2_xg,
    )

    row = {
        "match_id": event["id"],
        "commence_time": event["commence_time"],
        "team_1": team_1,
        "team_2": team_2,
        "team_1_xg": team_1_xg,
        "team_2_xg": team_2_xg,
        "expected_total_goals": fit["expected_total_goals"],
        "fit_loss": fit["fit_loss"],
        "h2h_bookmaker_count": h2h_bookmaker_count,
        "total_goal_line_count": total_goal_line_count(total_goal_probs),
        "total_goal_reference_points": total_goal_reference_points(total_goal_probs),
        "bookmaker_team_1_win": bookmaker_probs[team_1],
        "model_team_1_win": model_outcomes["win"],
        "diff_team_1_win": model_outcomes["win"] - bookmaker_probs[team_1],
        "bookmaker_draw": bookmaker_probs["Draw"],
        "model_draw": model_outcomes["draw"],
        "diff_draw": model_outcomes["draw"] - bookmaker_probs["Draw"],
        "bookmaker_team_2_win": bookmaker_probs[team_2],
        "model_team_2_win": model_outcomes["loss"],
        "diff_team_2_win": model_outcomes["loss"] - bookmaker_probs[team_2],
    }

    for total_probs in total_goal_probs:
        point = total_probs["point"]
        label = total_goal_point_column_label(point)
        model_totals = model_total_goal_probs(
            expected_total_goals=fit["expected_total_goals"],
            point=point,
        )

        row[f"total_goal_{label}_bookmaker_count"] = total_probs["bookmaker_count"]
        row[f"bookmaker_over_{label}"] = total_probs["Over"]
        row[f"model_over_{label}"] = model_totals["Over"]
        row[f"diff_over_{label}"] = model_totals["Over"] - total_probs["Over"]
        row[f"bookmaker_under_{label}"] = total_probs["Under"]
        row[f"model_under_{label}"] = model_totals["Under"]
        row[f"diff_under_{label}"] = model_totals["Under"] - total_probs["Under"]

    return row


def order_validation_columns(
    validation: pd.DataFrame,
    total_goal_points: set[float],
) -> pd.DataFrame:
    base_columns = [
        "match_id",
        "commence_time",
        "team_1",
        "team_2",
        "team_1_xg",
        "team_2_xg",
        "expected_total_goals",
        "fit_loss",
        "h2h_bookmaker_count",
        "total_goal_line_count",
        "total_goal_reference_points",
        "bookmaker_team_1_win",
        "model_team_1_win",
        "diff_team_1_win",
        "bookmaker_draw",
        "model_draw",
        "diff_draw",
        "bookmaker_team_2_win",
        "model_team_2_win",
        "diff_team_2_win",
    ]
    total_goal_columns = []

    for point in sorted(total_goal_points):
        label = total_goal_point_column_label(point)
        total_goal_columns.extend(
            [
                f"total_goal_{label}_bookmaker_count",
                f"bookmaker_over_{label}",
                f"model_over_{label}",
                f"diff_over_{label}",
                f"bookmaker_under_{label}",
                f"model_under_{label}",
                f"diff_under_{label}",
            ]
        )

    columns = [
        column
        for column in base_columns + total_goal_columns
        if column in validation.columns
    ]

    return validation[columns]


def parse_match_odds(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    xg = []
    validation = []
    total_goal_points = set()

    for event in events:
        team_1 = normalise_team(event["home_team"])
        team_2 = normalise_team(event["away_team"])

        bookmaker_probs = consensus_h2h_probs(event)

        p_team_1_win = bookmaker_probs[team_1]
        p_draw = bookmaker_probs["Draw"]
        p_team_2_win = bookmaker_probs[team_2]

        total_goal_probs = consensus_total_goal_probs(event)
        total_goal_points.update(probs["point"] for probs in total_goal_probs)
        fit = fit_market_implied_xg(
            p_team_1_win=p_team_1_win,
            p_draw=p_draw,
            p_team_2_win=p_team_2_win,
            total_goal_probs=total_goal_probs,
        )
        team_1_xg = fit["lambda_1"]
        team_2_xg = fit["lambda_2"]

        xg.extend(
            xg_rows(
                event=event,
                team_1=team_1,
                team_2=team_2,
                team_1_xg=team_1_xg,
                team_2_xg=team_2_xg,
            )
        )

        validation.append(
            validation_row(
                event=event,
                team_1=team_1,
                team_2=team_2,
                bookmaker_probs=bookmaker_probs,
                total_goal_probs=total_goal_probs,
                fit=fit,
                h2h_bookmaker_count=len(extract_bookmaker_h2h_probs(event)),
            )
        )

    validation_df = (
        pd.DataFrame(validation).sort_values("commence_time").reset_index(drop=True)
    )

    return (
        sort_match_rows(pd.DataFrame(xg)),
        order_validation_columns(
            validation=validation_df,
            total_goal_points=total_goal_points,
        ),
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

    xg_output, validation_output = parse_match_odds(events)
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(raw_path),
    )

    xg_output_path = dated_paths.input_csv_dir / "group_match_outcome_xg.csv"
    validation_output_path = (
        dated_paths.output_csv_dir / "group_match_odds_xg_validation.csv"
    )
    xg_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output_path.parent.mkdir(parents=True, exist_ok=True)

    xg_output.to_csv(xg_output_path, index=False)
    validation_output.to_csv(validation_output_path, index=False)

    text_output = "\n".join(
        [
            f"Read raw JSON from {raw_path}",
            f"Wrote xG CSV to {xg_output_path}",
            f"Wrote validation CSV to {validation_output_path}",
            xg_output.to_string(index=False),
            "",
            f"Number of fixtures: {xg_output['match_id'].nunique()}",
            f"Number of team-match xG rows: {len(xg_output)}",
            f"Number of validation rows: {len(validation_output)}",
        ]
    )
    click.echo(text_output)


if __name__ == "__main__":
    main()
