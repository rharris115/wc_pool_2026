import pandas as pd

try:
    from .paths import CSV_DIR
except ImportError:
    from paths import CSV_DIR

try:
    from .common import (
        ENTRANTS,
        MEDALS,
        find_latest_file,
        format_teams,
        format_whatsapp_table,
        load_probabilities,
        validate_teams,
    )
except ImportError:
    from common import (
        ENTRANTS,
        MEDALS,
        find_latest_file,
        format_teams,
        format_whatsapp_table,
        load_probabilities,
        validate_teams,
)

TEAM_WIN_PROBS_PATTERN = "team_win_probs_*.csv"


def calculate_first_prize_odds() -> pd.DataFrame:
    team_win_probs_file = find_latest_file(
        CSV_DIR,
        TEAM_WIN_PROBS_PATTERN,
    )

    prob_map = load_probabilities(team_win_probs_file)
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
                "teams": format_teams(teams),
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
    df = calculate_first_prize_odds()

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
