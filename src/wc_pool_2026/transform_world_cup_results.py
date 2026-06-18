from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.paths import (
    build_dated_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    dated_snapshot_dirs,
    find_dated_file,
    find_latest_dated_file,
    load_json,
    normalise_team,
    require_columns,
    snapshot_date_stamp as infer_snapshot_date_stamp,
    sort_match_rows,
)

RESULT_COLUMNS = [
    "match_id",
    "commence_time",
    "team",
    "opponent",
    "win",
    "draw",
    "loss",
    "team_g",
    "opponent_g",
    "total_goals",
]
RESULT_KEY_COLUMNS = ["match_id", "team"]


def find_latest_raw_file(resources_path: Path) -> Path:
    return find_latest_dated_file(
        resources_path=resources_path,
        subdirectory="raw_api",
        filename="world_cup_scores.json",
    )


def find_raw_file(
    resources_path: Path,
    date_stamp: str | None = None,
) -> Path:
    return find_dated_file(
        resources_path=resources_path,
        subdirectory="raw_api",
        filename="world_cup_scores.json",
        date_stamp=date_stamp,
    )


def score_map(event: dict) -> dict[str, int]:
    scores = event.get("scores")

    if not scores:
        raise ValueError(f"Missing scores for match_id={event['id']}")

    return {normalise_team(score["name"]): int(score["score"]) for score in scores}


def outcome_probs(
    team_goals: int,
    opponent_goals: int,
) -> tuple[int, int, int]:
    if team_goals > opponent_goals:
        return 1, 0, 0

    if team_goals == opponent_goals:
        return 0, 1, 0

    return 0, 0, 1


def result_row(
    event: dict,
    team: str,
    opponent: str,
    team_goals: int,
    opponent_goals: int,
) -> dict:
    p_win, p_draw, p_loss = outcome_probs(
        team_goals=team_goals,
        opponent_goals=opponent_goals,
    )
    total_goals = team_goals + opponent_goals

    return {
        "match_id": event["id"],
        "commence_time": event["commence_time"],
        "team": team,
        "opponent": opponent,
        "win": p_win,
        "draw": p_draw,
        "loss": p_loss,
        "team_g": team_goals,
        "opponent_g": opponent_goals,
        "total_goals": total_goals,
    }


def parse_results(events: list[dict]) -> pd.DataFrame:
    completed_events = sorted(
        (event for event in events if event.get("completed") and event.get("scores")),
        key=lambda event: event["commence_time"],
    )

    rows = []

    for event in completed_events:
        team_1 = normalise_team(event["home_team"])
        team_2 = normalise_team(event["away_team"])
        scores = score_map(event)

        missing_scores = {team_1, team_2} - set(scores)
        if missing_scores:
            raise ValueError(
                f"Missing scores for match_id={event['id']}: {missing_scores}"
            )

        team_1_goals = scores[team_1]
        team_2_goals = scores[team_2]

        rows.append(
            result_row(
                event=event,
                team=team_1,
                opponent=team_2,
                team_goals=team_1_goals,
                opponent_goals=team_2_goals,
            )
        )
        rows.append(
            result_row(
                event=event,
                team=team_2,
                opponent=team_1,
                team_goals=team_2_goals,
                opponent_goals=team_1_goals,
            )
        )

    return sort_match_rows(pd.DataFrame(rows, columns=RESULT_COLUMNS))


def previous_results_file(resources_path: Path, date_stamp: str) -> Path | None:
    previous_files = [
        file
        for snapshot_dir in dated_snapshot_dirs(resources_path)
        if snapshot_dir.name < date_stamp
        if (file := snapshot_dir / "input_csv" / "group_match_results.csv").is_file()
    ]

    return previous_files[-1] if previous_files else None


def load_previous_results(resources_path: Path, date_stamp: str) -> pd.DataFrame:
    previous_file = previous_results_file(
        resources_path=resources_path,
        date_stamp=date_stamp,
    )

    if previous_file is None:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    previous = pd.read_csv(previous_file)
    require_columns(
        df=previous,
        columns=set(RESULT_COLUMNS),
        source=previous_file,
    )

    return previous[RESULT_COLUMNS]


def disagreement_warnings(
    previous_results: pd.DataFrame,
    latest_results: pd.DataFrame,
) -> list[str]:
    if previous_results.empty or latest_results.empty:
        return []

    previous_by_key = previous_results.set_index(RESULT_KEY_COLUMNS)
    latest_by_key = latest_results.set_index(RESULT_KEY_COLUMNS)
    shared_index = previous_by_key.index.intersection(latest_by_key.index)
    compared_columns = [
        column for column in RESULT_COLUMNS if column not in RESULT_KEY_COLUMNS
    ]
    warnings = []

    for key in shared_index:
        previous_row = previous_by_key.loc[key]
        latest_row = latest_by_key.loc[key]
        differences = [
            (
                f"{column}: previous={previous_row[column]!r}, "
                f"latest={latest_row[column]!r}"
            )
            for column in compared_columns
            if previous_row[column] != latest_row[column]
        ]

        if differences:
            match_id, team = key
            warnings.append(
                f"match_id={match_id}, team={team}: " + "; ".join(differences)
            )

    return warnings


def merge_results(
    previous_results: pd.DataFrame,
    latest_results: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = disagreement_warnings(
        previous_results=previous_results,
        latest_results=latest_results,
    )
    merged = pd.concat(
        [previous_results, latest_results],
        ignore_index=True,
    )

    if merged.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), warnings

    merged = (
        merged.drop_duplicates(
            subset=RESULT_KEY_COLUMNS,
            keep="last",
        )
        .loc[:, RESULT_COLUMNS]
        .reset_index(drop=True)
    )

    return sort_match_rows(merged), warnings


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
    resources_path = resources_path.expanduser().resolve()
    raw_path = find_raw_file(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp,
    )
    events = load_json(raw_path)

    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp or infer_snapshot_date_stamp(raw_path),
    )
    date_stamp = dated_paths.snapshot_dir.name
    latest_results = parse_results(events)
    previous_results = load_previous_results(
        resources_path=resources_path,
        date_stamp=date_stamp,
    )
    output, warnings = merge_results(
        previous_results=previous_results,
        latest_results=latest_results,
    )

    output_path = dated_paths.input_csv_dir / "group_match_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    text_output = "\n".join(
        (
            [
                *(
                    [
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                        "WARNING: group_match_results.csv disagreement detected",
                        "The latest raw scores disagree with carried-forward results.",
                        *warnings,
                        "Latest raw scores have been used for the affected rows.",
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                        "",
                    ]
                    if warnings
                    else []
                ),
            f"Read raw JSON from {raw_path}",
            f"Carried forward {len(previous_results)} team-match rows",
            f"Parsed {len(latest_results)} latest team-match rows",
            f"Wrote CSV to {output_path}",
            output.to_string(index=False),
            "",
            f"Number of completed fixtures: {output['match_id'].nunique()}",
            f"Number of team-match rows: {len(output)}",
            ]
        )
    )
    click.echo(text_output)


if __name__ == "__main__":
    main()
