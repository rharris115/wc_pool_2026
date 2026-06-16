from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    MEDALS,
    PoolResources,
    find_match_xg_file,
    format_teams_with_metric,
    format_whatsapp_table,
    load_pool_resources,
    outcome_probs_from_xg,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    write_text_output,
)

N_SIMULATIONS = 10000000
RANDOM_SEED = 42
TIEBREAKER_CHUNK_SIZE = 250000


def find_match_results_file(
    match_xg_file: Path,
    resources_path: Path,
) -> Path:
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=infer_snapshot_date_stamp(match_xg_file),
    )
    match_results_file = dated_paths.input_csv_dir / "group_match_results.csv"

    if not match_results_file.is_file():
        raise FileNotFoundError(
            "No matching results CSV found for "
            f"{match_xg_file.name}: expected {match_results_file}"
        )

    return match_results_file


def load_fixtures(match_xg_file: Path) -> pd.DataFrame:
    rows = pd.read_csv(match_xg_file)

    required_columns = {
        "match_id",
        "commence_time",
        "team",
        "opponent",
        "team_xg",
        "opponent_xg",
    }

    require_columns(
        df=rows,
        columns=required_columns,
        source=match_xg_file,
    )

    fixtures = []

    for match_id, match_rows in rows.groupby("match_id", sort=False):
        if len(match_rows) != 2:
            raise ValueError(
                f"Expected 2 rows for match_id={match_id}, got {len(match_rows)}"
            )

        row = match_rows.iloc[0]
        reverse_row = match_rows.iloc[1]

        if (
            row["team"] != reverse_row["opponent"]
            or row["opponent"] != reverse_row["team"]
        ):
            raise ValueError(
                f"Expected reciprocal team rows for match_id={match_id}, got "
                f"{row['team']} vs {row['opponent']} and "
                f"{reverse_row['team']} vs {reverse_row['opponent']}"
            )

        if not np.isclose(row["team_xg"], reverse_row["opponent_xg"]) or not np.isclose(
            row["opponent_xg"],
            reverse_row["team_xg"],
        ):
            raise ValueError(
                f"Expected reciprocal xG rows for match_id={match_id}, got "
                f"{row['team_xg']}-{row['opponent_xg']} and "
                f"{reverse_row['team_xg']}-{reverse_row['opponent_xg']}"
            )

        outcome_probs = outcome_probs_from_xg(
            team_xg=row["team_xg"],
            opponent_xg=row["opponent_xg"],
        )

        fixtures.append(
            {
                "match_id": match_id,
                "commence_time": row["commence_time"],
                "team_1": row["team"],
                "team_2": row["opponent"],
                "p_team_1_win": outcome_probs["win"],
                "p_draw": outcome_probs["draw"],
                "p_team_2_win": outcome_probs["loss"],
                "team_1_xg": row["team_xg"],
                "team_2_xg": row["opponent_xg"],
            }
        )

    return pd.DataFrame(fixtures).reset_index(drop=True)


def load_completed_results(match_results_file: Path) -> pd.DataFrame:
    result_columns = [
        "match_id",
        "commence_time",
        "team_1",
        "team_2",
        "team_1_goals",
        "team_2_goals",
    ]
    rows = pd.read_csv(match_results_file)

    required_columns = {
        "match_id",
        "commence_time",
        "team",
        "opponent",
        "team_g",
        "opponent_g",
    }

    require_columns(
        df=rows,
        columns=required_columns,
        source=match_results_file,
    )

    results = []

    for match_id, match_rows in rows.groupby("match_id", sort=False):
        if len(match_rows) != 2:
            raise ValueError(
                f"Expected 2 rows for match_id={match_id}, got {len(match_rows)}"
            )

        row = match_rows.iloc[0]

        results.append(
            {
                "match_id": match_id,
                "commence_time": row["commence_time"],
                "team_1": row["team"],
                "team_2": row["opponent"],
                "team_1_goals": row["team_g"],
                "team_2_goals": row["opponent_g"],
            }
        )

    return pd.DataFrame(results, columns=result_columns).reset_index(drop=True)


def apply_completed_results(
    completed_results: pd.DataFrame,
    team_index: dict[str, int],
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
) -> None:
    for result in completed_results.to_dict("records"):
        team_1_index = team_index[result["team_1"]]
        team_2_index = team_index[result["team_2"]]

        team_1_goals = result["team_1_goals"]
        team_2_goals = result["team_2_goals"]

        goals_for[:, team_1_index] += team_1_goals
        goals_against[:, team_1_index] += team_2_goals
        goals_for[:, team_2_index] += team_2_goals
        goals_against[:, team_2_index] += team_1_goals

        if team_1_goals > team_2_goals:
            points[:, team_1_index] += 3
        elif team_1_goals < team_2_goals:
            points[:, team_2_index] += 3
        else:
            points[:, team_1_index] += 1
            points[:, team_2_index] += 1


def remaining_fixtures(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
) -> pd.DataFrame:
    if completed_results.empty:
        return fixtures

    completed_match_ids = set(completed_results["match_id"])

    return fixtures[~fixtures["match_id"].isin(completed_match_ids)].reset_index(
        drop=True
    )


def apply_worst_team_tiebreakers(
    points: np.ndarray,
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
    chunk_size: int = TIEBREAKER_CHUNK_SIZE,
) -> np.ndarray:
    worst_counts = np.zeros(points.shape[1], dtype=np.float64)
    sentinel = np.iinfo(np.int16).max

    for start in tqdm(
        range(0, points.shape[0], chunk_size),
        desc="Applying worst-team tiebreakers",
        unit="chunk",
    ):
        stop = min(start + chunk_size, points.shape[0])
        chunk_points = points[start:stop]
        chunk_goals_for = goals_for[start:stop]
        chunk_goal_difference = goal_difference[start:stop]

        min_points = chunk_points.min(axis=1)
        point_candidates = chunk_points == min_points[:, np.newaxis]

        candidate_goal_difference = np.where(
            point_candidates,
            chunk_goal_difference,
            sentinel,
        )
        min_goal_difference = candidate_goal_difference.min(axis=1)
        goal_difference_candidates = point_candidates & (
            chunk_goal_difference == min_goal_difference[:, np.newaxis]
        )

        candidate_goals_for = np.where(
            goal_difference_candidates,
            chunk_goals_for,
            sentinel,
        )
        min_goals_for = candidate_goals_for.min(axis=1)
        worst_candidates = goal_difference_candidates & (
            chunk_goals_for == min_goals_for[:, np.newaxis]
        )

        tie_counts = worst_candidates.sum(axis=1)
        worst_counts += (worst_candidates / tie_counts[:, np.newaxis]).sum(axis=0)

    return worst_counts


def simulate_group_stage(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    teams = sorted(
        set(fixtures["team_1"])
        | set(fixtures["team_2"])
        | set(completed_results["team_1"])
        | set(completed_results["team_2"])
    )
    team_index = {team: index for index, team in enumerate(teams)}

    points = np.zeros((n_simulations, len(teams)), dtype=np.int16)
    goals_for = np.zeros((n_simulations, len(teams)), dtype=np.int16)
    goals_against = np.zeros((n_simulations, len(teams)), dtype=np.int16)

    apply_completed_results(
        completed_results=completed_results,
        team_index=team_index,
        points=points,
        goals_for=goals_for,
        goals_against=goals_against,
    )

    rng = np.random.default_rng(random_seed)

    remaining_fixture_records = remaining_fixtures(
        fixtures=fixtures,
        completed_results=completed_results,
    ).to_dict("records")

    for fixture in tqdm(
        remaining_fixture_records,
        desc="Simulating fixtures",
        unit="match",
    ):
        team_1_index = team_index[fixture["team_1"]]
        team_2_index = team_index[fixture["team_2"]]

        team_1_goals = rng.poisson(
            fixture["team_1_xg"],
            size=n_simulations,
        )
        team_2_goals = rng.poisson(
            fixture["team_2_xg"],
            size=n_simulations,
        )

        goals_for[:, team_1_index] += team_1_goals
        goals_against[:, team_1_index] += team_2_goals
        goals_for[:, team_2_index] += team_2_goals
        goals_against[:, team_2_index] += team_1_goals

        team_1_wins = team_1_goals > team_2_goals
        team_2_wins = team_2_goals > team_1_goals
        draws = team_1_goals == team_2_goals

        points[team_1_wins, team_1_index] += 3
        points[team_2_wins, team_2_index] += 3
        points[draws, team_1_index] += 1
        points[draws, team_2_index] += 1

    goal_difference = goals_for - goals_against
    worst_counts = apply_worst_team_tiebreakers(
        points=points,
        goals_for=goals_for,
        goal_difference=goal_difference,
    )

    rows = []

    for team, index in team_index.items():
        expected_goals_for = goals_for[:, index].mean()
        expected_goals_against = goals_against[:, index].mean()

        rows.append(
            {
                "team": team,
                "worst_probability": worst_counts[index] / n_simulations,
                "expected_points": points[:, index].mean(),
                "expected_goals_for": expected_goals_for,
                "expected_goals_against": expected_goals_against,
                "expected_goal_difference": (
                    expected_goals_for - expected_goals_against
                ),
            }
        )

    team_results = (
        pd.DataFrame(rows)
        .sort_values("worst_probability", ascending=False)
        .reset_index(drop=True)
    )

    return team_results


def build_worst_probability_map(
    team_results: pd.DataFrame,
) -> dict[str, float]:
    return dict(
        zip(
            team_results["team"],
            team_results["worst_probability"],
        )
    )


def format_teams_with_worst_probability(
    teams: list[str],
    worst_probability_map: dict[str, float],
    resources: PoolResources,
) -> str:
    worst_probability_pct_map = {
        team: probability * 100 for team, probability in worst_probability_map.items()
    }

    return format_teams_with_metric(
        teams=teams,
        metric_map=worst_probability_pct_map,
        metric_format="{:.2f}%",
        team_emojis=resources.team_emojis,
    )


def build_entrant_leaderboard(
    team_results: pd.DataFrame,
    resources: PoolResources,
) -> pd.DataFrame:
    worst_probability_map = build_worst_probability_map(team_results)
    rows = []

    for person, teams in resources.entrants.items():
        probability = sum(worst_probability_map[team] for team in teams)

        rows.append(
            {
                "person": person,
                "probability": probability,
                "probability_pct": f"{probability * 100:.2f}%",
                "teams": format_teams_with_worst_probability(
                    teams=teams,
                    worst_probability_map=worst_probability_map,
                    resources=resources,
                ),
            }
        )

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    leaderboard.insert(
        0,
        "rank",
        [MEDALS.get(i + 1, str(i + 1)) for i in range(len(leaderboard))],
    )

    return leaderboard[
        [
            "rank",
            "person",
            "probability_pct",
            "teams",
        ]
    ]


def calculate_third_prize_monte_carlo(
    resources_path: Path,
    resources: PoolResources,
    date_stamp: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    match_xg_file = find_match_xg_file(
        resources_path=resources_path,
        date_stamp=date_stamp,
    )
    match_results_file = find_match_results_file(
        match_xg_file=match_xg_file,
        resources_path=resources_path,
    )
    fixtures = load_fixtures(match_xg_file)
    completed_results = load_completed_results(match_results_file)
    team_results = simulate_group_stage(
        fixtures=fixtures,
        completed_results=completed_results,
    )
    leaderboard = build_entrant_leaderboard(
        team_results=team_results,
        resources=resources,
    )

    return leaderboard, team_results, match_xg_file


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
    resources = load_pool_resources(resources_path)
    (
        leaderboard,
        team_results,
        match_xg_file,
    ) = calculate_third_prize_monte_carlo(
        resources_path=resources_path,
        resources=resources,
        date_stamp=snapshot_date_stamp,
    )

    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(match_xg_file),
    )
    team_output_path = dated_paths.output_csv_dir / "third_prize_monte_carlo_teams.csv"
    entrant_output_path = (
        dated_paths.output_csv_dir / "third_prize_monte_carlo_entrants.csv"
    )

    dated_paths.output_csv_dir.mkdir(parents=True, exist_ok=True)
    team_results.to_csv(team_output_path, index=False)
    leaderboard.to_csv(entrant_output_path, index=False)

    message = format_whatsapp_table(
        df=leaderboard,
        title="🥉 WORLD CUP SWEEPSTAKE – 3RD PRIZE MONTE CARLO",
        subtitle=(
            "Criterion: Your teams include the worst group-stage team, "
            "ranked by lowest points, worst goal difference, then fewest "
            "goals for. "
            "Ties are split equally."
        ),
    )

    text_output_path = (
        dated_paths.output_txt_dir / "calculate_third_prize_monte_carlo.txt"
    )
    write_text_output(
        path=text_output_path,
        text=message,
    )
    click.echo(f"Wrote team Monte Carlo CSV to {team_output_path}")
    click.echo(f"Wrote entrant Monte Carlo CSV to {entrant_output_path}")
    click.echo(f"Wrote WhatsApp text to {text_output_path}")


if __name__ == "__main__":
    main()
