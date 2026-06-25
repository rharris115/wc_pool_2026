from pathlib import Path
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from wc_pool_2026.common import (
    outcome_probs_from_xg,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
)
from wc_pool_2026.paths import build_dated_resource_paths

GROUP_LETTERS = tuple("ABCDEFGHIJKL")


@dataclass(frozen=True)
class SimulatedGroupMatch:
    match_id: str
    team_1_index: int
    team_2_index: int
    team_1_goals: int | np.ndarray
    team_2_goals: int | np.ndarray


@dataclass(frozen=True)
class GroupStageSimulation:
    teams: list[str]
    team_index: dict[str, int]
    points: np.ndarray
    goals_for: np.ndarray
    goals_against: np.ndarray
    matches: list[SimulatedGroupMatch]


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


def infer_groups(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
) -> dict[str, list[str]]:
    match_rows = pd.concat(
        [
            fixtures[["commence_time", "team_1", "team_2"]],
            completed_results[["commence_time", "team_1", "team_2"]],
        ],
        ignore_index=True,
    )
    teams = sorted(set(match_rows["team_1"]) | set(match_rows["team_2"]))
    edges = {
        frozenset((row["team_1"], row["team_2"]))
        for row in match_rows.to_dict("records")
    }

    group_candidates = [
        frozenset(candidate)
        for candidate in combinations(teams, 4)
        if all(frozenset(pair) in edges for pair in combinations(candidate, 2))
    ]
    groups = find_group_cover(teams=teams, group_candidates=group_candidates)
    first_match_time_by_group = {
        group: min(
            row["commence_time"]
            for row in match_rows.to_dict("records")
            if row["team_1"] in group and row["team_2"] in group
        )
        for group in groups
    }

    return {
        group_letter: sorted(group)
        for group_letter, group in zip(
            GROUP_LETTERS,
            sorted(groups, key=lambda group: first_match_time_by_group[group]),
            strict=True,
        )
    }


def find_group_cover(
    teams: list[str],
    group_candidates: list[frozenset[str]],
) -> list[frozenset[str]]:
    target_team_count = len(GROUP_LETTERS) * 4
    if len(teams) != target_team_count:
        raise ValueError(
            f"Expected {target_team_count} teams for group-stage inference, "
            f"got {len(teams)}"
        )

    candidates_by_team = {
        team: [candidate for candidate in group_candidates if team in candidate]
        for team in teams
    }

    def search(uncovered: frozenset[str]) -> list[frozenset[str]] | None:
        if not uncovered:
            return []

        seed = min(
            uncovered,
            key=lambda team: sum(
                candidate <= uncovered for candidate in candidates_by_team[team]
            ),
        )

        for candidate in candidates_by_team[seed]:
            if not candidate <= uncovered:
                continue

            remainder = search(uncovered - candidate)
            if remainder is not None:
                return [candidate, *remainder]

        return None

    groups = search(frozenset(teams))
    if groups is None or len(groups) != len(GROUP_LETTERS):
        raise ValueError(
            "Could not infer 12 four-team groups from complete internal "
            "group-stage match cliques"
        )

    return groups


def filter_group_stage_matches(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_by_team = {
        team: group
        for group, group_teams in groups.items()
        for team in group_teams
    }

    def group_match_mask(rows: pd.DataFrame) -> pd.Series:
        return rows.apply(
            lambda row: group_by_team[row["team_1"]] == group_by_team[row["team_2"]],
            axis=1,
        )

    return (
        fixtures[group_match_mask(fixtures)].reset_index(drop=True),
        completed_results[group_match_mask(completed_results)].reset_index(drop=True),
    )


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


def simulate_group_stats(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    n_simulations: int,
    random_seed: int,
    progress_description: str = "Simulating fixtures",
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    simulation = simulate_group_stage_state(
        fixtures=fixtures,
        completed_results=completed_results,
        n_simulations=n_simulations,
        random_seed=random_seed,
        progress_description=progress_description,
        record_matches=False,
    )

    return (
        simulation.teams,
        simulation.team_index,
        simulation.points,
        simulation.goals_for,
        simulation.goals_against,
    )


def simulate_group_stage_state(
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame,
    n_simulations: int,
    random_seed: int,
    progress_description: str = "Simulating fixtures",
    record_matches: bool = False,
) -> GroupStageSimulation:
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
    matches = []

    apply_completed_results(
        completed_results=completed_results,
        team_index=team_index,
        points=points,
        goals_for=goals_for,
        goals_against=goals_against,
    )

    if record_matches:
        for result in completed_results.to_dict("records"):
            matches.append(
                SimulatedGroupMatch(
                    match_id=str(result["match_id"]),
                    team_1_index=team_index[result["team_1"]],
                    team_2_index=team_index[result["team_2"]],
                    team_1_goals=int(result["team_1_goals"]),
                    team_2_goals=int(result["team_2_goals"]),
                )
            )

    rng = np.random.default_rng(random_seed)
    remaining_fixture_records = remaining_fixtures(
        fixtures=fixtures,
        completed_results=completed_results,
    ).to_dict("records")

    for fixture in tqdm(
        remaining_fixture_records,
        desc=progress_description,
        unit="match",
    ):
        team_1_index = team_index[fixture["team_1"]]
        team_2_index = team_index[fixture["team_2"]]

        team_1_goals = rng.poisson(
            fixture["team_1_xg"],
            size=n_simulations,
        ).astype(np.int16)
        team_2_goals = rng.poisson(
            fixture["team_2_xg"],
            size=n_simulations,
        ).astype(np.int16)

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

        if record_matches:
            matches.append(
                SimulatedGroupMatch(
                    match_id=str(fixture["match_id"]),
                    team_1_index=team_1_index,
                    team_2_index=team_2_index,
                    team_1_goals=team_1_goals,
                    team_2_goals=team_2_goals,
                )
            )

    return GroupStageSimulation(
        teams=teams,
        team_index=team_index,
        points=points,
        goals_for=goals_for,
        goals_against=goals_against,
        matches=matches,
    )
