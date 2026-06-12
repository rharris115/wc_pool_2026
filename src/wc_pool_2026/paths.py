from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourcePaths:
    config_dir: Path
    input_csv_dir: Path
    output_csv_dir: Path
    raw_api_dir: Path


def default_resources_path() -> Path:
    return Path(__file__).resolve().parent / ".." / ".." / "resources"


def build_resource_paths(resources_path: str | Path) -> ResourcePaths:
    config_dir = Path(resources_path).expanduser().resolve()

    return ResourcePaths(
        config_dir=config_dir,
        input_csv_dir=config_dir / "input_csv",
        output_csv_dir=config_dir / "output_csv",
        raw_api_dir=config_dir / "raw_api",
    )
