from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .paths import INPUT_CSV_DIR, OUTPUT_CSV_DIR
except ImportError:
    from paths import INPUT_CSV_DIR, OUTPUT_CSV_DIR

try:
    from .common import (
        ENTRANTS,
        MEDALS,
        TEAM_EMOJIS,
        extract_date_stamp,
        find_latest_file,
        format_whatsapp_table,
    )
except ImportError:
    from common import (
        ENTRANTS,
        MEDALS,
        TEAM_EMOJIS,
        extract_date_stamp,
        find_latest_file,
        format_whatsapp_table,
    )

MATCH_PROBS_PATTERN = "group_match_outcome_probs_*.csv"

N_SIMULATIONS = 1000000
RANDOM_SEED = 42


def find_latest_match_probs_file() -> Path:
    return find_latest_file(INPUT_CSV_DIR, MATCH_PROBS_PATTERN)


def build_team_output_path(match_probs_file: Path) -> Path:
    date_stamp = extract_date_stamp(match_probs_file)
    return OUTPUT_CSV_DIR / f"third_prize_monte_carlo_teams_{date_stamp}.csv"


def build_entrant_output_path(match_probs_file: Path) -> Path:
    date_stamp = extract_date_stamp(match_probs_file)
    return OUTPUT_CSV_DIR / f"third_prize_monte_carlo_entrants_{date_stamp}.csv"


def build_match_output_path(match_probs_file: Path) -> Path:
    date_stamp = extract_date_stamp(match_probs_file)
    return OUTPUT_CSV_DIR / f"third_prize_monte_carlo_matches_{date_stamp}.csv"


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
                "match_number": row["match_number"],
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

    return (
        pd.DataFrame(fixtures)
        .sort_values("commence_time")
        .reset_index(drop=True)
    )


def simulate_group_stage(
    fixtures: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = sorted(set(fixtures["team_1"]) | set(fixtures["team_2"]))
    team_index = {team: index for index, team in enumerate(teams)}

    points = np.zeros((n_simulations, len(teams)), dtype=np.int16)
    goals_for = np.zeros((n_simulations, len(teams)), dtype=np.int16)
    goals_against = np.zeros((n_simulations, len(teams)), dtype=np.int16)

    rng = np.random.default_rng(random_seed)

    fixture_records = fixtures.to_dict("records")
    match_rows = []

    for fixture in tqdm(
        fixture_records,
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
                "team_1": format_team_name(fixture["team_1"]),
                "team_2": format_team_name(fixture["team_2"]),
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
        simulation_goals_against = goals_against[simulation_index]
        simulation_goal_difference = goal_difference[simulation_index]

        candidates = simulation_points == simulation_points.min()
        candidates &= (
            simulation_goals_against
            == simulation_goals_against[candidates].max()
        )
        candidates &= (
            simulation_goal_difference
            == simulation_goal_difference[candidates].min()
        )
        candidates &= (
            simulation_goals_for
            == simulation_goals_for[candidates].min()
        )

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
                    expected_goals_for
                    - expected_goals_against
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


def format_team_name(team: str) -> str:
    return f"{TEAM_EMOJIS.get(team, '🏳️')} {team}"


def format_teams_with_worst_probability(
    teams: list[str],
    worst_probability_map: dict[str, float],
) -> str:
    return " | ".join(
        (
            f"{format_team_name(team)} "
            f"({worst_probability_map[team] * 100:.2f}%)"
        )
        for team in teams
    )


def format_team_results_table(team_results: pd.DataFrame) -> pd.DataFrame:
    output = team_results.copy()

    output["team"] = output["team"].map(format_team_name)
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
) -> pd.DataFrame:
    worst_probability_map = build_worst_probability_map(team_results)
    rows = []

    for person, teams in ENTRANTS.items():
        probability = sum(worst_probability_map[team] for team in teams)

        rows.append(
            {
                "person": person,
                "probability": probability,
                "probability_pct": f"{probability * 100:.2f}%",
                "teams": format_teams_with_worst_probability(
                    teams=teams,
                    worst_probability_map=worst_probability_map,
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

    return leaderboard[
        [
            "rank",
            "person",
            "probability_pct",
            "teams",
        ]
    ]


def calculate_third_prize_monte_carlo() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    match_probs_file = find_latest_match_probs_file()
    fixtures = load_fixtures(match_probs_file)
    team_results, match_results = simulate_group_stage(fixtures)
    leaderboard = build_entrant_leaderboard(team_results)

    return leaderboard, team_results, match_results, match_probs_file


def main() -> None:
    (
        leaderboard,
        team_results,
        match_results,
        match_probs_file,
    ) = calculate_third_prize_monte_carlo()

    team_output_path = build_team_output_path(match_probs_file)
    entrant_output_path = build_entrant_output_path(match_probs_file)
    match_output_path = build_match_output_path(match_probs_file)

    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    team_results.to_csv(team_output_path, index=False)
    leaderboard.to_csv(entrant_output_path, index=False)
    match_results.to_csv(match_output_path, index=False)

    print()
    print("🎲 TEAM MONTE CARLO THIRD PRIZE PROBABILITIES")
    print("=" * 120)
    print(
        format_team_results_table(
            team_results
        ).to_string(index=False)
    )

    print("\n")
    print(f"Wrote team Monte Carlo CSV to {team_output_path}")
    print(f"Wrote entrant Monte Carlo CSV to {entrant_output_path}")
    print(f"Wrote match Monte Carlo CSV to {match_output_path}")
    print("\n")

    message = format_whatsapp_table(
        df=leaderboard,
        title="💀 WORLD CUP SWEEPSTAKE – 3RD PRIZE MONTE CARLO",
        subtitle=(
            "Criterion: probability of owning the worst group-stage team "
            "after simulated scores, ranked by lowest points, most goals "
            "against, worst goal difference, then fewest goals for."
        ),
    )

    print(message)


if __name__ == "__main__":
    main()
