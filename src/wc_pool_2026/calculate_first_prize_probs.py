import pandas as pd
from pathlib import Path

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
        load_probabilities,
        validate_teams,
    )
except ImportError:
    from common import (
        ENTRANTS,
        MEDALS,
        TEAM_EMOJIS,
        extract_date_stamp,
        find_latest_file,
        format_whatsapp_table,
        load_probabilities,
        validate_teams,
)

WORLD_CUP_WINNER_ODDS_PATTERN = "world_cup_winner_odds_*.csv"


def build_entrant_output_path(world_cup_winner_odds_file: Path) -> Path:
    date_stamp = extract_date_stamp(world_cup_winner_odds_file)
    return OUTPUT_CSV_DIR / f"first_prize_entrants_{date_stamp}.csv"


def format_teams_with_win_probability(
    teams: list[str],
    prob_map: dict[str, float],
) -> str:
    return " ".join(
        f"{TEAM_EMOJIS.get(team, '🏳️')} {team} ({prob_map[team] * 100:.2f}%)"
        for team in teams
    )


def calculate_first_prize_odds() -> pd.DataFrame:
    world_cup_winner_odds_file = find_latest_file(
        INPUT_CSV_DIR,
        WORLD_CUP_WINNER_ODDS_PATTERN,
    )

    return build_first_prize_leaderboard(world_cup_winner_odds_file)


def build_first_prize_leaderboard(
    world_cup_winner_odds_file: Path,
) -> pd.DataFrame:
    prob_map = load_probabilities(world_cup_winner_odds_file)
    validate_teams(prob_map)

    rows = []

    for person, teams in ENTRANTS.items():
        probability_pct = (
            sum(
                prob_map[team]
                for team in teams
            )
            * 100
        )

        rows.append(
            {
                "person": person,
                "probability_pct": (
                    f"{probability_pct:.2f}%"
                ),
                "teams": format_teams_with_win_probability(
                    teams=teams,
                    prob_map=prob_map,
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "probability_pct",
            ascending=False,
            key=lambda s: (
                s.str.rstrip("%").astype(float)
            ),
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        [
            MEDALS.get(i + 1, str(i + 1))
            for i in range(len(result))
        ],
    )

    return result


def main() -> None:
    world_cup_winner_odds_file = find_latest_file(
        INPUT_CSV_DIR,
        WORLD_CUP_WINNER_ODDS_PATTERN,
    )
    df = build_first_prize_leaderboard(world_cup_winner_odds_file)
    entrant_output_path = build_entrant_output_path(world_cup_winner_odds_file)

    entrant_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(entrant_output_path, index=False)

    print(f"Wrote first prize entrant CSV to {entrant_output_path}")
    print()

    message = format_whatsapp_table(
        df=df,
        title="🏆 WORLD CUP SWEEPSTAKE – 1ST PRIZE ODDS",
        subtitle=(
            "Criterion: probability that one of your teams "
            "wins the 2026 FIFA World Cup"
        ),
    )

    print(message)


if __name__ == "__main__":
    main()
