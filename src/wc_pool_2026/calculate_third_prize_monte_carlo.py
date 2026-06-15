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
    find_match_probs_file,
    format_team_name,
    format_teams_with_metric,
    format_whatsapp_table,
    load_pool_resources,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
)

N_SIMULATIONS = 10000000
RANDOM_SEED = 42


def find_match_results_file(
    match_probs_file: Path,
    resources_path: Path,
) -> Path:
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=infer_snapshot_date_stamp(match_probs_file),
    )
    match_results_file = dated_paths.input_csv_dir / "group_match_results.csv"

    if not match_results_file.is_file():
        raise FileNotFoundError(
            "No matching results CSV found for "
            f"{match_probs_file.name}: expected {match_results_file}"
        )

    return match_results_file


def load_fixtures(match_probs_file: Path) -> pd.DataFrame:
    rows = pd.read_csv(match_probs_file)

    required_columns = {
        "match_id",
        "commence_time",
        "team",
        "opponent",
        "p_win",
        "p_draw",
        "p_loss",
        "team_xg",
        "opponent_xg",
    }

    require_columns(
        df=rows,
        columns=required_columns,
        source=match_probs_file,
    )

    fixtures = []

    for match_id, match_rows in rows.groupby("match_id"):
        if len(match_rows) != 2:
            raise ValueError(
                f"Expected 2 rows for match_id={match_id}, got {len(match_rows)}"
            )

        row = match_rows.iloc[0]

        fixtures.append(
            {
                "match_id": match_id,
                "commence_time": row["commence_time"],
                "team_1": row["team"],
                "team_2": row["opponent"],
                "p_team_1_win": row["p_win"],
                "p_draw": row["p_draw"],
                "p_team_2_win": row["p_loss"],
                "team_1_xg": row["team_xg"],
                "team_2_xg": row["opponent_xg"],
            }
        )

    return pd.DataFrame(fixtures).sort_values("commence_time").reset_index(drop=True)


def load_completed_results(match_results_file: Path) -> pd.DataFrame:
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

    for match_id, match_rows in rows.groupby("match_id"):
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

    return pd.DataFrame(results).sort_values("commence_time").reset_index(drop=True)


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


def simulate_group_stage(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    resources: PoolResources,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    match_rows = []

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

        simulated_team_1_win = team_1_wins.mean()
        simulated_draw = draws.mean()
        simulated_team_2_win = team_2_wins.mean()

        match_rows.append(
            {
                "team_1": format_team_name(
                    fixture["team_1"],
                    resources.team_emojis,
                ),
                "team_2": format_team_name(
                    fixture["team_2"],
                    resources.team_emojis,
                ),
                "team_1_win": fixture["p_team_1_win"],
                "simulated_team_1_win": simulated_team_1_win,
                "draw": fixture["p_draw"],
                "simulated_draw": simulated_draw,
                "team_2_win": fixture["p_team_2_win"],
                "simulated_team_2_win": simulated_team_2_win,
            }
        )

        points[team_1_wins, team_1_index] += 3
        points[team_2_wins, team_2_index] += 3
        points[draws, team_1_index] += 1
        points[draws, team_2_index] += 1

    goal_difference = goals_for - goals_against
    worst_counts = np.zeros(len(teams), dtype=np.float64)

    for simulation_index in tqdm(
        range(n_simulations),
        desc="Applying worst-team tiebreakers",
        unit="sim",
    ):
        simulation_points = points[simulation_index]
        simulation_goals_for = goals_for[simulation_index]
        simulation_goal_difference = goal_difference[simulation_index]

        candidates = simulation_points == simulation_points.min()
        candidates &= (
            simulation_goal_difference == simulation_goal_difference[candidates].min()
        )
        candidates &= simulation_goals_for == simulation_goals_for[candidates].min()

        worst_counts[candidates] += 1 / candidates.sum()

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
    match_results = pd.DataFrame(match_rows)

    return team_results, match_results


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


def format_team_results_table(
    team_results: pd.DataFrame,
    resources: PoolResources,
) -> pd.DataFrame:
    output = team_results.copy()

    output["team"] = output["team"].map(
        lambda team: format_team_name(team, resources.team_emojis)
    )
    output["worst_probability_pct"] = output["worst_probability"].map(
        lambda value: f"{value * 100:.2f}%"
    )

    for column in [
        "expected_points",
        "expected_goals_for",
        "expected_goals_against",
        "expected_goal_difference",
    ]:
        output[column] = output[column].map(lambda value: f"{value:.2f}")

    return output[
        [
            "team",
            "worst_probability_pct",
            "expected_points",
            "expected_goals_for",
            "expected_goals_against",
            "expected_goal_difference",
        ]
    ]


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
    pd.DataFrame,
    Path,
]:
    match_probs_file = find_match_probs_file(
        resources_path=resources_path,
        date_stamp=date_stamp,
    )
    match_results_file = find_match_results_file(
        match_probs_file=match_probs_file,
        resources_path=resources_path,
    )
    fixtures = load_fixtures(match_probs_file)
    completed_results = load_completed_results(match_results_file)
    team_results, match_results = simulate_group_stage(
        fixtures=fixtures,
        completed_results=completed_results,
        resources=resources,
    )
    leaderboard = build_entrant_leaderboard(
        team_results=team_results,
        resources=resources,
    )

    return leaderboard, team_results, match_results, match_probs_file


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
        match_results,
        match_probs_file,
    ) = calculate_third_prize_monte_carlo(
        resources_path=resources_path,
        resources=resources,
        date_stamp=snapshot_date_stamp,
    )

    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(match_probs_file),
    )
    team_output_path = dated_paths.output_csv_dir / "third_prize_monte_carlo_teams.csv"
    entrant_output_path = (
        dated_paths.output_csv_dir / "third_prize_monte_carlo_entrants.csv"
    )
    match_output_path = (
        dated_paths.output_csv_dir / "third_prize_monte_carlo_matches.csv"
    )

    dated_paths.output_csv_dir.mkdir(parents=True, exist_ok=True)
    team_results.to_csv(team_output_path, index=False)
    leaderboard.to_csv(entrant_output_path, index=False)
    match_results.to_csv(match_output_path, index=False)

    print()
    print("🎲 TEAM MONTE CARLO THIRD PRIZE PROBABILITIES")
    print("=" * 120)
    print(
        format_team_results_table(
            team_results=team_results,
            resources=resources,
        ).to_string(index=False)
    )

    print("\n")
    print(f"Wrote team Monte Carlo CSV to {team_output_path}")
    print(f"Wrote entrant Monte Carlo CSV to {entrant_output_path}")
    print(f"Wrote match Monte Carlo CSV to {match_output_path}")
    print("\n")

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

    print(message)


if __name__ == "__main__":
    main()
