from pathlib import Path

import click
import numpy as np
import pandas as pd

from wc_pool_2026.common import (
    MEDALS,
    PoolResources,
    find_match_xg_file,
    format_teams_with_metric,
    format_whatsapp_table,
    load_pool_resources,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    write_text_output,
)
from wc_pool_2026.group_stage_monte_carlo import (
    filter_group_stage_matches,
    find_match_results_file,
    infer_groups,
    load_completed_results,
    load_fixtures,
    simulate_group_stats,
)
from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)

N_SIMULATIONS = 10000000
RANDOM_SEED = 42


def apply_worst_team_tiebreakers(
    points: np.ndarray,
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
) -> np.ndarray:
    worst_counts = np.zeros(points.shape[1], dtype=np.float64)
    sentinel = np.iinfo(np.int16).max

    min_points = points.min(axis=1)
    point_candidates = points == min_points[:, np.newaxis]
    point_tie_counts = point_candidates.sum(axis=1)
    unique_worst_on_points = point_candidates & (point_tie_counts[:, np.newaxis] == 1)
    worst_counts += unique_worst_on_points.sum(axis=0)

    candidate_goal_difference = np.where(
        point_candidates,
        goal_difference,
        sentinel,
    )
    min_goal_difference = candidate_goal_difference.min(axis=1)
    goal_difference_candidates = point_candidates & (
        goal_difference == min_goal_difference[:, np.newaxis]
    )
    goal_difference_tie_counts = goal_difference_candidates.sum(axis=1)
    unique_worst_on_goal_difference = (
        goal_difference_candidates
        & (point_tie_counts[:, np.newaxis] > 1)
        & (goal_difference_tie_counts[:, np.newaxis] == 1)
    )
    worst_counts += unique_worst_on_goal_difference.sum(axis=0)

    candidate_goals_for = np.where(
        goal_difference_candidates,
        goals_for,
        sentinel,
    )
    min_goals_for = candidate_goals_for.min(axis=1)
    worst_candidates = goal_difference_candidates & (
        goals_for == min_goals_for[:, np.newaxis]
    )
    goals_for_tie_counts = worst_candidates.sum(axis=1)
    unique_worst_on_goals_for = (
        worst_candidates
        & (goal_difference_tie_counts[:, np.newaxis] > 1)
        & (goals_for_tie_counts[:, np.newaxis] == 1)
    )
    worst_counts += unique_worst_on_goals_for.sum(axis=0)

    unresolved_ties = (
        worst_candidates
        & (goal_difference_tie_counts[:, np.newaxis] > 1)
        & (goals_for_tie_counts[:, np.newaxis] > 1)
    )
    simulation_indexes, team_indexes = np.nonzero(unresolved_ties)
    np.add.at(
        worst_counts,
        team_indexes,
        1 / goals_for_tie_counts[simulation_indexes],
    )

    return worst_counts


def simulate_group_stage(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    (
        teams,
        team_index,
        points,
        goals_for,
        goals_against,
    ) = simulate_group_stats(
        fixtures=fixtures,
        completed_results=completed_results,
        n_simulations=n_simulations,
        random_seed=random_seed,
        progress_description="Simulating fixtures",
    )

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
    groups = infer_groups(
        fixtures=fixtures,
        completed_results=completed_results,
    )
    fixtures, completed_results = filter_group_stage_matches(
        fixtures=fixtures,
        completed_results=completed_results,
        groups=groups,
    )
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
