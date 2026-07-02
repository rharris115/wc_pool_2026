from pathlib import Path
import re

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from wc_pool_2026.calculate_third_prize_monte_carlo import RANDOM_SEED
from wc_pool_2026.common import (
    find_match_xg_file,
    format_team_name,
    load_pool_resources,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    write_text_output,
)
from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.group_stage_monte_carlo import (
    SimulatedGroupMatch,
    filter_group_stage_matches,
    find_match_results_file,
    load_completed_results,
    load_fixtures,
    load_groups,
    simulate_group_stage_state,
)

GROUP_LETTERS = tuple("ABCDEFGHIJKL")
GROUP_SLOT_PATTERN = re.compile(r"^([123])([A-L])$")
N_SIMULATIONS = 1
THIRD_PLACE_CRITERIA_NOTE = (
    "Third-place ranking: points, goal difference, then goals scored. "
    "Any remaining ties are split randomly because team conduct scores and "
    "FIFA ranking tie-break data are not currently available in the input CSVs. "
    "Group-position ties use points, then head-to-head points, head-to-head "
    "goal difference, and head-to-head goals scored between the tied teams. "
    "If teams remain tied, overall goal difference and overall goals scored are "
    "used. Remaining ties are split randomly because conduct scores and FIFA "
    "ranking data are not currently available in the input CSVs."
)


def load_bracket(path: Path) -> pd.DataFrame:
    bracket = pd.read_csv(path, dtype={"match_id": int, "winner_to": "Int64"})
    require_columns(
        df=bracket,
        columns={"match_id", "round", "home_source", "away_source", "winner_to"},
        source=path,
    )

    if bracket["match_id"].duplicated().any():
        raise ValueError(f"Duplicate match_id values in {path}")

    return bracket.sort_values("match_id").reset_index(drop=True)


def load_third_place_assignments(path: Path) -> dict[str, dict[int, str]]:
    assignments = pd.read_csv(path)
    require_columns(
        df=assignments,
        columns={"qualifying_groups_key"},
        source=path,
    )

    match_columns = [
        column for column in assignments.columns if re.fullmatch(r"match_\d+", column)
    ]
    if not match_columns:
        raise ValueError(f"No match assignment columns found in {path}")

    rows = {}
    for row in assignments.to_dict("records"):
        key = row["qualifying_groups_key"]
        if not re.fullmatch(r"[A-L]{8}", key):
            raise ValueError(f"Invalid qualifying_groups_key in {path}: {key}")

        rows[key] = {
            int(column.removeprefix("match_")): row[column]
            for column in match_columns
        }

    if len(rows) != len(assignments):
        raise ValueError(f"Duplicate qualifying_groups_key values in {path}")

    return rows


def rank_group_indexes(
    group_indexes: list[int],
    points: np.ndarray,
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
    rng: np.random.Generator,
    group_matches: list[SimulatedGroupMatch],
) -> np.ndarray:
    group_indexes_array = np.array(group_indexes)
    random_tiebreaker = rng.random((points.shape[0], len(group_indexes)))
    rankings = np.empty((points.shape[0], len(group_indexes)), dtype=np.int16)

    for simulation_index in range(points.shape[0]):
        rankings[simulation_index] = rank_group_for_simulation(
            group_indexes=list(group_indexes_array),
            points=points,
            goals_for=goals_for,
            goal_difference=goal_difference,
            group_matches=group_matches,
            random_tiebreaker=random_tiebreaker[simulation_index],
            simulation_index=simulation_index,
        )

    return rankings


def goals_for_simulation(goals: int | np.ndarray, simulation_index: int) -> int:
    if isinstance(goals, np.ndarray):
        return int(goals[simulation_index])
    return goals


def rank_group_for_simulation(
    group_indexes: list[int],
    points: np.ndarray,
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
    group_matches: list[SimulatedGroupMatch],
    random_tiebreaker: np.ndarray,
    simulation_index: int,
) -> list[int]:
    point_groups: dict[int, list[int]] = {}

    for team_index in group_indexes:
        point_groups.setdefault(int(points[simulation_index, team_index]), []).append(
            team_index
        )

    ranked = []
    for point_total in sorted(point_groups, reverse=True):
        tied_team_indexes = point_groups[point_total]
        if len(tied_team_indexes) == 1:
            ranked.extend(tied_team_indexes)
        else:
            ranked.extend(
                rank_tied_teams(
                    tied_team_indexes=tied_team_indexes,
                    all_group_indexes=group_indexes,
                    points=points,
                    goals_for=goals_for,
                    goal_difference=goal_difference,
                    group_matches=group_matches,
                    random_tiebreaker=random_tiebreaker,
                    simulation_index=simulation_index,
                )
            )

    return ranked


def rank_tied_teams(
    tied_team_indexes: list[int],
    all_group_indexes: list[int],
    points: np.ndarray,
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
    group_matches: list[SimulatedGroupMatch],
    random_tiebreaker: np.ndarray,
    simulation_index: int,
) -> list[int]:
    head_to_head_groups: dict[tuple[int, int, int], list[int]] = {}

    for team_index in tied_team_indexes:
        head_to_head_groups.setdefault(
            head_to_head_key(
                team_index=team_index,
                tied_team_indexes=tied_team_indexes,
                group_matches=group_matches,
                simulation_index=simulation_index,
            ),
            [],
        ).append(team_index)

    if len(head_to_head_groups) > 1:
        ranked = []
        for key in sorted(head_to_head_groups, reverse=True):
            still_tied = head_to_head_groups[key]
            if len(still_tied) == 1:
                ranked.extend(still_tied)
            else:
                ranked.extend(
                    rank_tied_teams(
                        tied_team_indexes=still_tied,
                        all_group_indexes=all_group_indexes,
                        points=points,
                        goals_for=goals_for,
                        goal_difference=goal_difference,
                        group_matches=group_matches,
                        random_tiebreaker=random_tiebreaker,
                        simulation_index=simulation_index,
                    )
                )
        return ranked

    return sorted(
        tied_team_indexes,
        key=lambda team_index: overall_sort_key(
            team_index=team_index,
            all_group_indexes=all_group_indexes,
            goals_for=goals_for,
            goal_difference=goal_difference,
            random_tiebreaker=random_tiebreaker,
            simulation_index=simulation_index,
        ),
        reverse=True,
    )


def head_to_head_key(
    team_index: int,
    tied_team_indexes: list[int],
    group_matches: list[SimulatedGroupMatch],
    simulation_index: int,
) -> tuple[int, int, int]:
    tied_team_index_set = set(tied_team_indexes)
    head_to_head_points = 0
    head_to_head_goals_for = 0
    head_to_head_goals_against = 0

    for match in group_matches:
        if (
            match.team_1_index not in tied_team_index_set
            or match.team_2_index not in tied_team_index_set
        ):
            continue

        team_1_goals = goals_for_simulation(
            goals=match.team_1_goals,
            simulation_index=simulation_index,
        )
        team_2_goals = goals_for_simulation(
            goals=match.team_2_goals,
            simulation_index=simulation_index,
        )

        if team_index == match.team_1_index:
            team_goals = team_1_goals
            opponent_goals = team_2_goals
        elif team_index == match.team_2_index:
            team_goals = team_2_goals
            opponent_goals = team_1_goals
        else:
            continue

        head_to_head_goals_for += team_goals
        head_to_head_goals_against += opponent_goals

        if team_goals > opponent_goals:
            head_to_head_points += 3
        elif team_goals == opponent_goals:
            head_to_head_points += 1

    return (
        head_to_head_points,
        head_to_head_goals_for - head_to_head_goals_against,
        head_to_head_goals_for,
    )


def overall_sort_key(
    team_index: int,
    all_group_indexes: list[int],
    goals_for: np.ndarray,
    goal_difference: np.ndarray,
    random_tiebreaker: np.ndarray,
    simulation_index: int,
) -> tuple[int, int, float]:
    local_index = all_group_indexes.index(team_index)

    return (
        int(goal_difference[simulation_index, team_index]),
        int(goals_for[simulation_index, team_index]),
        float(random_tiebreaker[local_index]),
    )


def build_group_standings(
    groups: dict[str, list[str]],
    team_index: dict[str, int],
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    matches: list[SimulatedGroupMatch],
    random_seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed + 1)
    goal_difference = goals_for - goals_against
    group_index_sets = {
        group: {team_index[team] for team in group_teams}
        for group, group_teams in groups.items()
    }
    group_matches = {
        group: [
            match
            for match in matches
            if match.team_1_index in group_index_sets[group]
            and match.team_2_index in group_index_sets[group]
        ]
        for group in groups
    }

    return {
        group: rank_group_indexes(
            group_indexes=[team_index[team] for team in group_teams],
            points=points,
            goals_for=goals_for,
            goal_difference=goal_difference,
            rng=rng,
            group_matches=group_matches[group],
        )
        for group, group_teams in groups.items()
    }


def third_place_keys(
    standings: dict[str, np.ndarray],
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed + 2)
    goal_difference = goals_for - goals_against
    third_indexes = np.column_stack(
        [standings[group][:, 2] for group in GROUP_LETTERS]
    )
    random_tiebreaker = rng.random(third_indexes.shape)
    keys = np.empty(points.shape[0], dtype=object)
    order = np.lexsort(
        (
            -random_tiebreaker,
            -goals_for[np.arange(points.shape[0])[:, np.newaxis], third_indexes],
            -goal_difference[np.arange(points.shape[0])[:, np.newaxis], third_indexes],
            -points[np.arange(points.shape[0])[:, np.newaxis], third_indexes],
        ),
        axis=1,
    )

    for simulation_index in range(points.shape[0]):
        qualifying_groups = sorted(
            GROUP_LETTERS[index] for index in order[simulation_index, :8]
        )
        keys[simulation_index] = "".join(qualifying_groups)

    return keys


def resolve_group_slot(
    source: str,
    standings: dict[str, np.ndarray],
) -> np.ndarray:
    match = GROUP_SLOT_PATTERN.fullmatch(source)
    if not match:
        raise ValueError(f"Invalid fixed group source: {source}")

    rank = int(match.group(1))
    group = match.group(2)
    return standings[group][:, rank - 1]


def resolve_source(
    source: str,
    match_id: int,
    standings: dict[str, np.ndarray],
    third_place_assignments: dict[str, dict[int, str]],
    qualifying_keys: np.ndarray,
) -> np.ndarray:
    if "|" in source:
        resolved = np.empty(len(qualifying_keys), dtype=np.int16)
        allowed_slots = set(source.split("|"))
        for key in np.unique(qualifying_keys):
            concrete_source = third_place_assignments[key][match_id]
            if concrete_source not in allowed_slots:
                raise ValueError(
                    f"Third-place assignment {concrete_source} is invalid for "
                    f"match {match_id} source {source}"
                )
            mask = qualifying_keys == key
            resolved[mask] = resolve_group_slot(
                source=concrete_source,
                standings=standings,
            )[mask]
        return resolved

    return resolve_group_slot(
        source=source,
        standings=standings,
    )


def add_slot_rows(
    rows: list[dict[str, object]],
    teams: list[str],
    match_id: int,
    round_name: str,
    slot: str,
    source: str,
    team_indexes: np.ndarray,
    n_simulations: int,
) -> None:
    unique_indexes, counts = np.unique(team_indexes, return_counts=True)

    for team_index, count in zip(unique_indexes, counts, strict=True):
        rows.append(
            {
                "match_id": match_id,
                "round": round_name,
                "slot": slot,
                "source": source,
                "team": teams[int(team_index)],
                "appearances": int(count),
                "probability": count / n_simulations,
            }
        )


def r32_bracket(bracket: pd.DataFrame) -> pd.DataFrame:
    r32 = bracket[bracket["round"] == "R32"].copy()
    if r32.empty:
        raise ValueError("Bracket does not contain any R32 rows")

    invalid_sources = sorted(
        {
            source
            for source in pd.concat([r32["home_source"], r32["away_source"]])
            if source.startswith("W")
        }
    )
    if invalid_sources:
        raise ValueError(
            "Round of 32 sources must come directly from group standings, got "
            f"{invalid_sources}"
        )

    return r32.sort_values("match_id").reset_index(drop=True)


def calculate_r32_slot_probabilities(
    bracket: pd.DataFrame,
    third_place_assignments: dict[str, dict[int, str]],
    standings: dict[str, np.ndarray],
    qualifying_keys: np.ndarray,
    teams: list[str],
    n_simulations: int,
) -> pd.DataFrame:
    slot_rows = []

    for match in tqdm(
        r32_bracket(bracket).to_dict("records"),
        desc="Resolving Round of 32 slots",
        unit="match",
    ):
        match_id = int(match["match_id"])
        home_source = match["home_source"]
        away_source = match["away_source"]
        home = resolve_source(
            source=home_source,
            match_id=match_id,
            standings=standings,
            third_place_assignments=third_place_assignments,
            qualifying_keys=qualifying_keys,
        )
        away = resolve_source(
            source=away_source,
            match_id=match_id,
            standings=standings,
            third_place_assignments=third_place_assignments,
            qualifying_keys=qualifying_keys,
        )

        add_slot_rows(
            rows=slot_rows,
            teams=teams,
            match_id=match_id,
            round_name=match["round"],
            slot="home",
            source=home_source,
            team_indexes=home,
            n_simulations=n_simulations,
        )
        add_slot_rows(
            rows=slot_rows,
            teams=teams,
            match_id=match_id,
            round_name=match["round"],
            slot="away",
            source=away_source,
            team_indexes=away,
            n_simulations=n_simulations,
        )

    return (
        pd.DataFrame(slot_rows)
        .sort_values(["match_id", "slot", "probability"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def build_third_place_key_results(
    qualifying_keys: np.ndarray,
    n_simulations: int,
) -> pd.DataFrame:
    keys, counts = np.unique(qualifying_keys, return_counts=True)
    return (
        pd.DataFrame(
            {
                "qualifying_groups_key": keys,
                "appearances": counts,
                "probability": counts / n_simulations,
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )


def format_r32_slot_summary(
    slot_results: pd.DataFrame,
    team_emojis: dict[str, str],
) -> str:
    lines = ["*WORLD CUP ROUND OF 32 SLOT MONTE CARLO*", ""]
    r32 = slot_results[slot_results["round"] == "R32"]

    for match_id, match_rows in r32.groupby("match_id", sort=True):
        lines.append(f"Match {match_id}")
        for slot in ("home", "away"):
            slot_rows = match_rows[match_rows["slot"] == slot].head(5)
            source = slot_rows.iloc[0]["source"]
            teams = ", ".join(
                f"{format_team_name(row.team, team_emojis)} "
                f"({row.probability * 100:.1f}%)"
                for row in slot_rows.itertuples()
            )
            lines.append(f"{slot} {source}: {teams}")
        lines.append("")

    lines.append(THIRD_PLACE_CRITERIA_NOTE)

    return "\n".join(lines).rstrip()


def calculate_r32_slot_monte_carlo(
    resources_path: Path,
    date_stamp: str | None,
    n_simulations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
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
    groups = load_groups(resources_path)
    fixtures, completed_results = filter_group_stage_matches(
        fixtures=fixtures,
        completed_results=completed_results,
        groups=groups,
    )
    simulation = simulate_group_stage_state(
        fixtures=fixtures,
        completed_results=completed_results,
        n_simulations=n_simulations,
        random_seed=random_seed,
        progress_description="Simulating group fixtures",
        record_matches=True,
    )
    standings = build_group_standings(
        groups=groups,
        team_index=simulation.team_index,
        points=simulation.points,
        goals_for=simulation.goals_for,
        goals_against=simulation.goals_against,
        matches=simulation.matches,
        random_seed=random_seed,
    )
    qualifying_keys = third_place_keys(
        standings=standings,
        points=simulation.points,
        goals_for=simulation.goals_for,
        goals_against=simulation.goals_against,
        random_seed=random_seed,
    )

    bracket = load_bracket(resources_path / "knockout_bracket.csv")
    third_place_assignments = load_third_place_assignments(
        resources_path / "third_place_assignment.csv"
    )
    missing_assignment_keys = sorted(set(qualifying_keys) - set(third_place_assignments))
    if missing_assignment_keys:
        raise ValueError(
            "Missing third-place assignment rows for keys "
            f"{missing_assignment_keys}"
        )

    slot_results = calculate_r32_slot_probabilities(
        bracket=bracket,
        third_place_assignments=third_place_assignments,
        standings=standings,
        qualifying_keys=qualifying_keys,
        teams=simulation.teams,
        n_simulations=n_simulations,
    )
    third_place_key_results = build_third_place_key_results(
        qualifying_keys=qualifying_keys,
        n_simulations=n_simulations,
    )

    return slot_results, third_place_key_results, match_xg_file


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
        "Dated resource snapshot to read/write, for example 20260624. "
        "Defaults to inferring the latest matching snapshot from resources."
    ),
)
@click.option(
    "--n-simulations",
    default=N_SIMULATIONS,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    "--random-seed",
    default=RANDOM_SEED,
    show_default=True,
    type=int,
)
def main(
    resources_path: Path,
    snapshot_date_stamp: str | None,
    n_simulations: int,
    random_seed: int,
) -> None:
    resources = load_pool_resources(resources_path)
    (
        slot_results,
        third_place_key_results,
        match_xg_file,
    ) = calculate_r32_slot_monte_carlo(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(match_xg_file),
    )
    dated_paths.output_csv_dir.mkdir(parents=True, exist_ok=True)

    slot_output_path = dated_paths.output_csv_dir / "r32_slot_monte_carlo_slots.csv"
    third_place_key_output_path = (
        dated_paths.output_csv_dir / "r32_slot_monte_carlo_third_place_keys.csv"
    )

    slot_results.to_csv(slot_output_path, index=False)
    third_place_key_results.to_csv(third_place_key_output_path, index=False)

    text_output_path = dated_paths.output_txt_dir / "calculate_r32_slot_monte_carlo.txt"
    write_text_output(
        path=text_output_path,
        text=format_r32_slot_summary(
            slot_results=slot_results,
            team_emojis=resources.team_emojis,
        ),
    )

    click.echo(f"Wrote Round of 32 slot CSV to {slot_output_path}")
    click.echo(f"Wrote third-place key CSV to {third_place_key_output_path}")
    click.echo(f"Wrote text summary to {text_output_path}")


if __name__ == "__main__":
    main()
