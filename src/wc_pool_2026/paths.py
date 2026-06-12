from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourcePaths:
    config_dir: Path


@dataclass(frozen=True)
class DatedResourcePaths:
    date_stamp: str
    snapshot_dir: Path
    input_csv_dir: Path
    output_csv_dir: Path
    raw_api_dir: Path


def default_resources_path() -> Path:
    return Path(__file__).resolve().parent / ".." / ".." / "resources"


def build_resource_paths(resources_path: str | Path) -> ResourcePaths:
    config_dir = Path(resources_path).expanduser().resolve()

    return ResourcePaths(
        config_dir=config_dir,
    )


def build_dated_resource_paths(
    resources: ResourcePaths,
    date_stamp: str,
) -> DatedResourcePaths:
    snapshot_dir = resources.config_dir / date_stamp

    return DatedResourcePaths(
        date_stamp=date_stamp,
        snapshot_dir=snapshot_dir,
        input_csv_dir=snapshot_dir / "input_csv",
        output_csv_dir=snapshot_dir / "output_csv",
        raw_api_dir=snapshot_dir / "raw_api",
    )
