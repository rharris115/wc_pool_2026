from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.paths import (
    ResourcePaths,
    build_resource_paths,
    default_resources_path,
)
from wc_pool_2026.common import (
    extract_date_stamp,
    find_latest_file,
    load_json,
    normalise_team,
    sort_match_rows,
)

def find_latest_raw_file(paths: ResourcePaths) -> Path:
    return find_latest_file(paths.raw_api_dir, "world_cup_scores_*.json")


def score_map(event: dict) -> dict[str, int]:
    scores = event.get("scores")

    if not scores:
        raise ValueError(f"Missing scores for match_id={event['id']}")

    return {
        normalise_team(score["name"]): int(score["score"])
        for score in scores
    }


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
        (
            event
            for event in events
            if event.get("completed") and event.get("scores")
        ),
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

    return sort_match_rows(pd.DataFrame(rows))


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
def main(resources_path: Path) -> None:
    paths = build_resource_paths(resources_path)
    raw_path = find_latest_raw_file(paths)
    events = load_json(raw_path)

    output = parse_results(events)

    output_path = (
        paths.input_csv_dir
        / f"group_match_results_{extract_date_stamp(raw_path)}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Read raw JSON from {raw_path}")
    print(f"Wrote CSV to {output_path}")
    print(output.to_string(index=False))
    print("\nNumber of completed fixtures:", output["match_id"].nunique())
    print("Number of team-match rows:", len(output))


if __name__ == "__main__":
    main()
