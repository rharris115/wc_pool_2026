import json
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re

import pandas as pd

MATCH_XG_FILE = "group_match_outcome_xg.csv"
MAX_GOALS_FOR_OUTCOME_PROBS = 12

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


def write_text_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text.rstrip()}\n", encoding="utf-8")


def load_pool_resources(config_dir: Path) -> PoolResources:
    return PoolResources(
        entrants=load_json(config_dir / "entrants.json"),
        team_emojis=load_json(config_dir / "team_emojis.json"),
    )


def normalise_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def entrant_teams(entrants: dict[str, list[str]]) -> set[str]:
    return {team for teams in entrants.values() for team in teams}


def current_date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def dated_snapshot_dirs(resources_path: Path) -> list[Path]:
    return sorted(
        path
        for path in resources_path.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}", path.name)
    )


def snapshot_date_stamp(path: Path) -> str:
    snapshot_dir = path.parent.parent

    if not re.fullmatch(r"\d{8}", snapshot_dir.name):
        raise ValueError(f"Could not determine dated snapshot directory for {path}")

    return snapshot_dir.name


def find_latest_dated_file(
    resources_path: Path,
    subdirectory: str,
    filename: str,
) -> Path:
    files = [
        file
        for snapshot_dir in dated_snapshot_dirs(resources_path)
        if (file := snapshot_dir / subdirectory / filename).is_file()
    ]

    if not files:
        raise FileNotFoundError(
            "No file named "
            f"{filename} found under dated {subdirectory} directories in "
            f"{resources_path.resolve()}"
        )

    return sorted(
        files,
        key=snapshot_date_stamp,
    )[-1]


def find_dated_file(
    resources_path: Path,
    subdirectory: str,
    filename: str,
    date_stamp: str | None = None,
) -> Path:
    if date_stamp is None:
        return find_latest_dated_file(
            resources_path=resources_path,
            subdirectory=subdirectory,
            filename=filename,
        )

    file = resources_path / date_stamp / subdirectory / filename

    if not file.is_file():
        raise FileNotFoundError(f"No file named {filename} found at {file}")

    return file


def find_latest_match_xg_file(resources_path: Path) -> Path:
    return find_match_xg_file(
        resources_path=resources_path,
        date_stamp=None,
    )


def find_match_xg_file(
    resources_path: Path,
    date_stamp: str | None = None,
) -> Path:
    return find_dated_file(
        resources_path=resources_path,
        subdirectory="input_csv",
        filename=MATCH_XG_FILE,
        date_stamp=date_stamp,
    )


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    source: Path,
) -> None:
    missing_columns = columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {source}: {missing_columns}")


def poisson_probs(expected_goals: float, max_goals: int) -> list[float]:
    probs = [math.exp(-expected_goals)]

    for goals in range(1, max_goals + 1):
        probs.append(probs[-1] * expected_goals / goals)

    return probs


def outcome_probs_from_xg(
    team_xg: float,
    opponent_xg: float,
    max_goals: int = MAX_GOALS_FOR_OUTCOME_PROBS,
) -> dict[str, float]:
    team_goal_probs = poisson_probs(team_xg, max_goals)
    opponent_goal_probs = poisson_probs(opponent_xg, max_goals)

    p_win = 0.0
    p_draw = 0.0
    p_loss = 0.0

    for team_goals, team_prob in enumerate(team_goal_probs):
        for opponent_goals, opponent_prob in enumerate(opponent_goal_probs):
            probability = team_prob * opponent_prob

            if team_goals > opponent_goals:
                p_win += probability
            elif team_goals == opponent_goals:
                p_draw += probability
            else:
                p_loss += probability

    total = p_win + p_draw + p_loss

    return {
        "win": p_win / total,
        "draw": p_draw / total,
        "loss": p_loss / total,
    }


def sort_match_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["commence_time", "team", "opponent"]).reset_index(drop=True)


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
        {team for teams in entrants.values() for team in teams if team not in prob_map}
    )

    if missing_teams:
        raise ValueError(f"Missing probabilities for teams: " f"{missing_teams}")


def format_teams(
    teams: list[str],
    team_emojis: dict[str, str],
) -> str:
    return " ".join(format_team_name(team, team_emojis) for team in teams)


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
