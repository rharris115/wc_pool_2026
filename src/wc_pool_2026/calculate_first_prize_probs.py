import pandas as pd
from pathlib import Path

import click

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    MEDALS,
    PoolResources,
    find_dated_file,
    format_teams_with_metric,
    format_whatsapp_table,
    load_pool_resources,
    load_probabilities,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    validate_teams,
)

WORLD_CUP_WINNER_ODDS_FILE = "world_cup_winner_odds.csv"


def calculate_first_prize_odds(
    resources_path: Path,
    resources: PoolResources,
    date_stamp: str | None = None,
) -> pd.DataFrame:
    world_cup_winner_odds_file = find_dated_file(
        resources_path=resources_path,
        subdirectory="input_csv",
        filename=WORLD_CUP_WINNER_ODDS_FILE,
        date_stamp=date_stamp,
    )

    return build_first_prize_leaderboard(
        world_cup_winner_odds_file=world_cup_winner_odds_file,
        resources=resources,
    )


def build_first_prize_leaderboard(
    world_cup_winner_odds_file: Path,
    resources: PoolResources,
) -> pd.DataFrame:
    prob_map = load_probabilities(world_cup_winner_odds_file)
    validate_teams(
        prob_map=prob_map,
        entrants=resources.entrants,
    )
    probability_pct_map = {
        team: probability * 100 for team, probability in prob_map.items()
    }

    rows = []

    for person, teams in resources.entrants.items():
        probability_pct = sum(prob_map[team] for team in teams) * 100

        rows.append(
            {
                "person": person,
                "probability_pct": (f"{probability_pct:.2f}%"),
                "teams": format_teams_with_metric(
                    teams=teams,
                    metric_map=probability_pct_map,
                    metric_format="{:.2f}%",
                    team_emojis=resources.team_emojis,
                    separator=" ",
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "probability_pct",
            ascending=False,
            key=lambda s: (s.str.rstrip("%").astype(float)),
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        [MEDALS.get(i + 1, str(i + 1)) for i in range(len(result))],
    )

    return result


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
    world_cup_winner_odds_file = find_dated_file(
        resources_path=resources_path,
        subdirectory="input_csv",
        filename=WORLD_CUP_WINNER_ODDS_FILE,
        date_stamp=snapshot_date_stamp,
    )
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp
        or infer_snapshot_date_stamp(world_cup_winner_odds_file),
    )
    df = build_first_prize_leaderboard(
        world_cup_winner_odds_file=world_cup_winner_odds_file,
        resources=resources,
    )
    entrant_output_path = dated_paths.output_csv_dir / "first_prize_entrants.csv"

    entrant_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(entrant_output_path, index=False)

    print(f"Wrote first prize entrant CSV to {entrant_output_path}")
    print()

    message = format_whatsapp_table(
        df=df,
        title="🥇 WORLD CUP SWEEPSTAKE – 1ST PRIZE ODDS",
        subtitle=("Criterion: One of your teams wins the 2026 FIFA World Cup."),
    )

    print(message)


if __name__ == "__main__":
    main()
