from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd

try:
    from .paths import CSV_DIR
except ImportError:
    from paths import CSV_DIR

try:
    from .common import (
        ENTRANTS,
        MEDALS,
        TEAM_EMOJIS,
        find_latest_file,
        format_whatsapp_table,
    )
except ImportError:
    from common import (
        ENTRANTS,
        MEDALS,
        TEAM_EMOJIS,
        find_latest_file,
        format_whatsapp_table,
    )

MATCH_PROBS_PATTERN = "group_match_outcome_probs_*.csv"

POSSIBLE_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def find_latest_match_probs_file() -> Path:
    return find_latest_file(CSV_DIR, MATCH_PROBS_PATTERN)


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

    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {match_probs_file}: {missing_columns}")

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
            }
        )

    return (
        pd.DataFrame(fixtures)
        .sort_values("commence_time")
        .reset_index(drop=True)
    )


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
            fixtures["team_1"].isin(component)
            & fixtures["team_2"].isin(component)
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
            team
            for team, team_points in points.items()
            if team_points == min_points
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
                points * point_probs[points]
                for points in POSSIBLE_POINTS
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


def format_team_name(team: str) -> str:
    return f"{TEAM_EMOJIS.get(team, '🏳️')} {team}"


def format_teams_with_expected_points(
    teams: list[str],
    expected_points_map: dict[str, float],
) -> str:
    return " | ".join(
        f"{format_team_name(team)} ({expected_points_map[team]:.2f})"
        for team in teams
    )


def format_points_distribution_table(
    distributions: pd.DataFrame,
) -> pd.DataFrame:
    output = distributions.copy()

    output["team"] = output["team"].map(format_team_name)

    for points in POSSIBLE_POINTS:
        col = f"{points}_pts"
        output[col] = output[col].mul(100).map(lambda value: f"{value:.1f}%")

    output["expected_points"] = output["expected_points"].map(
        lambda value: f"{value:.2f}"
    )

    return output


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


def calculate_third_prize_odds() -> tuple[pd.DataFrame, pd.DataFrame]:
    match_probs_file = find_latest_match_probs_file()

    fixtures = load_fixtures(match_probs_file)
    groups = assign_groups(fixtures)

    group_scenarios = [
        enumerate_group_scenarios(group)
        for group in groups
    ]

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

    for person, teams in ENTRANTS.items():
        probability = sum(team_worst_probs[team] for team in teams)

        rows.append(
            {
                "person": person,
                "probability": probability,
                "probability_pct": f"{probability * 100:.2f}%",
                "teams": format_teams_with_expected_points(
                    teams=teams,
                    expected_points_map=expected_points_map,
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
        [
            MEDALS.get(i + 1, str(i + 1))
            for i in range(len(leaderboard))
        ],
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


def main() -> None:
    leaderboard, points_distributions = calculate_third_prize_odds()

    print()
    print("📊 TEAM GROUP POINTS DISTRIBUTIONS")
    print("=" * 120)
    print(
        format_points_distribution_table(
            points_distributions
        ).to_string(index=False)
    )

    print("\n")

    message = format_whatsapp_table(
        df=leaderboard,
        title="💀 WORLD CUP SWEEPSTAKE – 3RD PRIZE ODDS",
        subtitle=(
            "Criterion: probability of owning the team with the lowest "
            "group-stage points. Equal split if teams tie on points."
        ),
    )

    print(message)


if __name__ == "__main__":
    main()
