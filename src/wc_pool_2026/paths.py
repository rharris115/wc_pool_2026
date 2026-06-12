from pathlib import Path


def find_project_root() -> Path:
    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ]

    for candidate in candidates:
        resources_dir = candidate / "resources"

        if (
            resources_dir.is_dir()
            and (resources_dir / "entrants.json").is_file()
            and (resources_dir / "team_emojis.json").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find project root containing resources/entrants.json "
        "and resources/team_emojis.json"
    )


PROJECT_ROOT = find_project_root()
CONFIG_DIR = PROJECT_ROOT / "resources"
CSV_DIR = CONFIG_DIR / "csv"
ENV_FILE = PROJECT_ROOT / ".env"
