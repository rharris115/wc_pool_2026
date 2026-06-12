import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from wc_pool_2026.paths import ResourcePaths

MATCH_PROBS_FILE = "group_match_outcome_probs.csv"

MEDALS = {
    1: "🥇",
    2: "🥈",
    3: "🥉",
}

TEAM_ALIASES = {
    "Turkey": "Türkiye",
    "Türkiye": "Türkiye",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia & Herzegovina": "Bosnia & Herzegovina",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "USA": "USA",
}


@dataclass(frozen=True)
class PoolResources:
    entrants: dict[str, list[str]]
    team_emojis: dict[str, str]


def load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_pool_resources(config_dir: Path) -> PoolResources:
    return PoolResources(
        entrants=load_json(config_dir / "entrants.json"),
        team_emojis=load_json(config_dir / "team_emojis.json"),
    )


def normalise_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def entrant_teams(entrants: dict[str, list[str]]) -> set[str]:
    return {
        team
        for teams in entrants.values()
        for team in teams
    }


def current_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def dated_snapshot_dirs(resources: ResourcePaths) -> list[Path]:
    return sorted(
        path
        for path in resources.config_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}", path.name)
    )


def snapshot_date_stamp(path: Path) -> str:
    snapshot_dir = path.parent.parent

    if not re.fullmatch(r"\d{8}", snapshot_dir.name):
        raise ValueError(
            f"Could not determine dated snapshot directory for {path}"
        )

    return snapshot_dir.name


def find_latest_dated_file(
    resources: ResourcePaths,
    subdirectory: str,
    filename: str,
) -> Path:
    files = [
        file
        for snapshot_dir in dated_snapshot_dirs(resources)
        if (file := snapshot_dir / subdirectory / filename).is_file()
    ]

    if not files:
        raise FileNotFoundError(
            "No file named "
            f"{filename} found under dated {subdirectory} directories in "
            f"{resources.config_dir.resolve()}"
        )

    return sorted(
        files,
        key=snapshot_date_stamp,
    )[-1]


def find_latest_match_probs_file(resources: ResourcePaths) -> Path:
    return find_latest_dated_file(
        resources=resources,
        subdirectory="input_csv",
        filename=MATCH_PROBS_FILE,
    )


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    missing_columns = columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {source}: {missing_columns}"
        )


def sort_match_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["commence_time", "team", "opponent"])
        .reset_index(drop=True)
    )


def load_probabilities(
    prob_file: Path,
) -> dict[str, float]:
    probs = pd.read_csv(prob_file)

    require_columns(
        df=probs,
        columns={"team", "win_prob"},
        source=prob_file,
    )

    return dict(
        zip(
            probs["team"],
            probs["win_prob"],
        )
    )


def validate_teams(
    prob_map: dict[str, float],
    entrants: dict[str, list[str]],
) -> None:
    missing_teams = sorted(
        {
            team
            for teams in entrants.values()
            for team in teams
            if team not in prob_map
        }
    )

    if missing_teams:
        raise ValueError(
            f"Missing probabilities for teams: "
            f"{missing_teams}"
        )


def format_teams(
    teams: list[str],
    team_emojis: dict[str, str],
) -> str:
    return " ".join(
        format_team_name(team, team_emojis)
        for team in teams
    )


def format_team_name(
    team: str,
    team_emojis: dict[str, str],
) -> str:
    return f"{team_emojis.get(team, '🏳️')} {team}"


def format_teams_with_metric(
    teams: list[str],
    metric_map: dict[str, float],
    metric_format: str,
    team_emojis: dict[str, str],
    separator: str = " | ",
) -> str:
    return separator.join(
        (
            f"{format_team_name(team, team_emojis)} "
            f"({metric_format.format(metric_map[team])})"
        )
        for team in teams
    )


def format_whatsapp_table(
    df,
    title: str,
    subtitle: str,
) -> str:
    lines = [
        f"*{title}*",
    ]

    for _, row in df.iterrows():
        lines.append(
            (
                f"{row['rank']} *{row['person']}* "
                f"({row['probability_pct']})\n"
                f"{row['teams']}"
            )
        )

    lines.append(subtitle)

    return "\n\n".join(lines)
