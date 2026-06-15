from collections import defaultdict
from itertools import product
from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    MEDALS,
    PoolResources,
    find_match_probs_file,
    format_teams_with_metric,
    format_whatsapp_table,
    load_pool_resources,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    write_text_output,
)

POSSIBLE_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9]


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
    }

    require_columns(
        df=rows,
        columns=required_columns,
        source=match_probs_file,
    )

    fixtures = []

    for match_id, match_rows in rows.groupby("match_id", sort=False):
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
            }
        )

    return pd.DataFrame(fixtures).reset_index(drop=True)


def assign_groups(fixtures: pd.DataFrame) -> list[dict]:
    teams = sorted(set(fixtures["team_1"]) | set(fixtures["team_2"]))

    graph = {team: set() for team in teams}

    for _, row in fixtures.iterrows():
        graph[row["team_1"]].add(row["team_2"])
        graph[row["team_2"]].add(row["team_1"])

    seen = set()
    groups = []

    for team in teams:
        if team in seen:
            continue

        stack = [team]
        component = set()

        while stack:
            current = stack.pop()

            if current in seen:
                continue

            seen.add(current)
            component.add(current)
            stack.extend(graph[current] - seen)

        group_fixtures = fixtures[
            fixtures["team_1"].isin(component) & fixtures["team_2"].isin(component)
        ].copy()

        groups.append(
            {
                "teams": sorted(component),
                "fixtures": group_fixtures,
            }
        )

    return groups


def enumerate_group_scenarios(group: dict) -> list[dict]:
    teams = group["teams"]
    fixtures = group["fixtures"].to_dict("records")

    scenarios = []

    for outcomes in product(["team_1", "draw", "team_2"], repeat=len(fixtures)):
        points = {team: 0 for team in teams}
        probability = 1.0

        for fixture, outcome in zip(fixtures, outcomes):
            team_1 = fixture["team_1"]
            team_2 = fixture["team_2"]

            if outcome == "team_1":
                points[team_1] += 3
                probability *= fixture["p_team_1_win"]
            elif outcome == "draw":
                points[team_1] += 1
                points[team_2] += 1
                probability *= fixture["p_draw"]
            elif outcome == "team_2":
                points[team_2] += 3
                probability *= fixture["p_team_2_win"]
            else:
                raise ValueError(f"Unknown outcome: {outcome}")

        min_points = min(points.values())
        min_teams = [
            team for team, team_points in points.items() if team_points == min_points
        ]

        scenarios.append(
            {
                "probability": probability,
                "points": points,
                "min_points": min_points,
                "min_teams": min_teams,
                "min_count": len(min_teams),
            }
        )

    return scenarios


def calculate_team_points_distributions(
    group_scenarios: list[list[dict]],
    groups: list[dict],
) -> pd.DataFrame:
    rows = []

    for group_index, scenarios in enumerate(group_scenarios):
        for team in groups[group_index]["teams"]:
            point_probs = {points: 0.0 for points in POSSIBLE_POINTS}

            for scenario in scenarios:
                team_points = scenario["points"][team]
                point_probs[team_points] += scenario["probability"]

            row = {"team": team}

            for points in POSSIBLE_POINTS:
                row[f"{points}_pts"] = point_probs[points]

            row["expected_points"] = sum(
                points * point_probs[points] for points in POSSIBLE_POINTS
            )

            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("expected_points", ascending=True)
        .reset_index(drop=True)
    )


def build_expected_points_map(
    points_distributions: pd.DataFrame,
) -> dict[str, float]:
    return dict(
        zip(
            points_distributions["team"],
            points_distributions["expected_points"],
        )
    )


def format_teams_with_expected_points(
    teams: list[str],
    expected_points_map: dict[str, float],
    resources: PoolResources,
) -> str:
    return format_teams_with_metric(
        teams=teams,
        metric_map=expected_points_map,
        metric_format="{:.2f}",
        team_emojis=resources.team_emojis,
    )


def count_distribution_for_other_groups(
    group_scenarios: list[list[dict]],
    excluded_group_index: int,
    points_level: int,
) -> dict[int, float]:
    distribution = {0: 1.0}

    for group_index, scenarios in enumerate(group_scenarios):
        if group_index == excluded_group_index:
            continue

        group_distribution = defaultdict(float)

        for scenario in scenarios:
            min_points = scenario["min_points"]

            if min_points < points_level:
                continue

            if min_points == points_level:
                count_at_level = scenario["min_count"]
            else:
                count_at_level = 0

            group_distribution[count_at_level] += scenario["probability"]

        new_distribution = defaultdict(float)

        for existing_count, existing_prob in distribution.items():
            for group_count, group_prob in group_distribution.items():
                new_distribution[existing_count + group_count] += (
                    existing_prob * group_prob
                )

        distribution = dict(new_distribution)

    return distribution


def calculate_team_worst_probabilities(
    group_scenarios: list[list[dict]],
    groups: list[dict],
) -> dict[str, float]:
    team_probs = defaultdict(float)

    for group_index, scenarios in enumerate(group_scenarios):
        for team in groups[group_index]["teams"]:
            for points_level in range(10):
                other_count_distribution = count_distribution_for_other_groups(
                    group_scenarios=group_scenarios,
                    excluded_group_index=group_index,
                    points_level=points_level,
                )

                for scenario in scenarios:
                    if scenario["min_points"] != points_level:
                        continue

                    if team not in scenario["min_teams"]:
                        continue

                    own_min_count = scenario["min_count"]

                    for other_count, other_prob in other_count_distribution.items():
                        team_probs[team] += (
                            scenario["probability"]
                            * other_prob
                            / (own_min_count + other_count)
                        )

    return dict(team_probs)


def calculate_third_prize_odds(
    resources_path: Path,
    resources: PoolResources,
    date_stamp: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_probs_file = find_match_probs_file(
        resources_path=resources_path,
        date_stamp=date_stamp,
    )

    fixtures = load_fixtures(match_probs_file)
    groups = assign_groups(fixtures)

    group_scenarios = [enumerate_group_scenarios(group) for group in groups]

    points_distributions = calculate_team_points_distributions(
        group_scenarios=group_scenarios,
        groups=groups,
    )

    expected_points_map = build_expected_points_map(points_distributions)

    team_worst_probs = calculate_team_worst_probabilities(
        group_scenarios=group_scenarios,
        groups=groups,
    )

    total_probability = sum(team_worst_probs.values())

    if abs(total_probability - 1.0) > 1e-6:
        raise RuntimeError(
            f"Team worst probabilities sum to {total_probability}, expected 1.0"
        )

    rows = []

    for person, teams in resources.entrants.items():
        probability = sum(team_worst_probs[team] for team in teams)

        rows.append(
            {
                "person": person,
                "probability": probability,
                "probability_pct": f"{probability * 100:.2f}%",
                "teams": format_teams_with_expected_points(
                    teams=teams,
                    expected_points_map=expected_points_map,
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

    return (
        leaderboard[
            [
                "rank",
                "person",
                "probability_pct",
                "teams",
            ]
        ],
        points_distributions,
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
    resources = load_pool_resources(resources_path)
    match_probs_file = find_match_probs_file(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp,
    )
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(match_probs_file),
    )
    leaderboard, points_distributions = calculate_third_prize_odds(
        resources_path=resources_path,
        resources=resources,
        date_stamp=snapshot_date_stamp,
    )

    message = format_whatsapp_table(
        df=leaderboard,
        title="🥉 WORLD CUP SWEEPSTAKE – 3RD PRIZE ODDS",
        subtitle=(
            "Criterion: Your teams include the team with the lowest "
            "group-stage points. Ties on points are split equally."
        ),
    )

    text_output_path = dated_paths.output_txt_dir / "calculate_third_prize_odds.txt"
    write_text_output(
        path=text_output_path,
        text=message,
    )
    click.echo(f"Wrote WhatsApp text to {text_output_path}")


if __name__ == "__main__":
    main()
