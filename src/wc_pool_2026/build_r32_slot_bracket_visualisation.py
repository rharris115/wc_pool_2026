from html import escape
import math
from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.common import (
    find_dated_file,
    format_team_name,
    load_pool_resources,
    require_columns,
    write_text_output,
)
from wc_pool_2026.paths import build_dated_resource_paths, default_resources_path
from wc_pool_2026.viz_common import (
    COPY_CHART_CONTROLS_CSS,
    COPY_CHART_PNG_SCRIPT,
    copy_chart_button,
)

HTML_FILE = "r32_slot_bracket.html"
SLOT_CSV_FILE = "r32_slot_monte_carlo_slots.csv"
BRACKET_CSV_FILE = "knockout_bracket.csv"

CARD_WIDTH = 320
MIN_CARD_HEIGHT = 226
MATCH_HEAD_HEIGHT = 22
SLOT_BASE_HEIGHT = 22
TEAM_ROW_HEIGHT = 15
CARD_VERTICAL_PADDING = 16
ROW_GAP = 22
MARGIN_X = 24
MARGIN_Y = 24
COLUMN_GAP = 62
SVG_WIDTH = MARGIN_X * 2 + CARD_WIDTH * 9 + COLUMN_GAP * 8
ELIMINATED_SECTION_TOP_GAP = 18
ELIMINATED_SECTION_BASE_HEIGHT = 70
ELIMINATED_ROW_HEIGHT = 30
ELIMINATED_CHIPS_PER_ROW = 8

LEFT_R32_MATCHES = [73, 74, 75, 76, 77, 78, 79, 80]
RIGHT_R32_MATCHES = [81, 82, 83, 84, 85, 86, 87, 88]
ROUND_LAYOUT = {
    89: (2, 1, 2),
    90: (2, 3, 2),
    91: (2, 5, 2),
    92: (2, 7, 2),
    97: (3, 1, 4),
    98: (3, 5, 4),
    101: (4, 1, 8),
    103: (5, 1, 8),
    102: (6, 1, 8),
    99: (7, 1, 4),
    100: (7, 5, 4),
    93: (8, 1, 2),
    94: (8, 3, 2),
    95: (8, 5, 2),
    96: (8, 7, 2),
}


def load_slot_probabilities(path: Path) -> pd.DataFrame:
    slots = pd.read_csv(path)
    require_columns(
        df=slots,
        columns={
            "match_id",
            "round",
            "slot",
            "source",
            "team",
            "appearances",
            "probability",
        },
        source=path,
    )
    return slots


def load_bracket(path: Path) -> pd.DataFrame:
    bracket = pd.read_csv(path)
    require_columns(
        df=bracket,
        columns={"match_id", "round", "home_source", "away_source", "winner_to"},
        source=path,
    )
    return bracket


def load_snapshot_teams(path: Path) -> list[str]:
    rows = pd.read_csv(path)
    require_columns(
        df=rows,
        columns={"team", "opponent"},
        source=path,
    )
    return sorted(set(rows["team"]) | set(rows["opponent"]))


def probability_label(probability: float) -> str:
    if probability >= 0.01:
        return f"{probability * 100:.1f}%"
    return f"{probability * 100:.2f}%"


def column_x(column: int) -> int:
    return MARGIN_X + (column - 1) * (CARD_WIDTH + COLUMN_GAP)


def r32_slot_row_count(slots: pd.DataFrame, match_id: int, slot_name: str) -> int:
    return len(slots[(slots["match_id"] == match_id) & (slots["slot"] == slot_name)])


def r32_card_height(slots: pd.DataFrame, match_id: int) -> int:
    team_rows = sum(
        r32_slot_row_count(slots=slots, match_id=match_id, slot_name=slot_name)
        for slot_name in ("home", "away")
    )
    content_height = (
        CARD_VERTICAL_PADDING
        + MATCH_HEAD_HEIGHT
        + SLOT_BASE_HEIGHT * 2
        + TEAM_ROW_HEIGHT * team_rows
    )
    return max(MIN_CARD_HEIGHT, content_height)


def row_heights(slots: pd.DataFrame) -> list[int]:
    return [
        max(
            r32_card_height(slots=slots, match_id=left_match_id),
            r32_card_height(slots=slots, match_id=right_match_id),
            MIN_CARD_HEIGHT,
        )
        for left_match_id, right_match_id in zip(
            LEFT_R32_MATCHES,
            RIGHT_R32_MATCHES,
            strict=True,
        )
    ]


def row_tops(heights: list[int]) -> list[int]:
    tops = []
    current_y = MARGIN_Y

    for height in heights:
        tops.append(current_y)
        current_y += height + ROW_GAP

    return tops


def centred_y(
    row: int,
    span: int,
    heights: list[int],
    tops: list[int],
    card_height: int = MIN_CARD_HEIGHT,
) -> int:
    group_top = tops[row - 1]
    group_bottom = tops[row - 1 + span - 1] + heights[row - 1 + span - 1]
    return round(group_top + (group_bottom - group_top - card_height) / 2)


def match_positions(
    slots: pd.DataFrame,
) -> tuple[dict[int, tuple[int, int]], dict[int, int], int]:
    heights = row_heights(slots)
    tops = row_tops(heights)
    positions = {}
    card_heights = {}

    for row_index, match_id in enumerate(LEFT_R32_MATCHES):
        card_heights[match_id] = r32_card_height(slots=slots, match_id=match_id)
        positions[match_id] = (column_x(1), tops[row_index])

    for row_index, match_id in enumerate(RIGHT_R32_MATCHES):
        card_heights[match_id] = r32_card_height(slots=slots, match_id=match_id)
        positions[match_id] = (column_x(9), tops[row_index])

    for match_id, (column, row, span) in ROUND_LAYOUT.items():
        card_heights[match_id] = MIN_CARD_HEIGHT
        positions[match_id] = (
            column_x(column),
            centred_y(row=row, span=span, heights=heights, tops=tops),
        )

    svg_height = MARGIN_Y * 2 + sum(heights) + ROW_GAP * (len(heights) - 1)
    return positions, card_heights, svg_height


def svg_text(
    x: float,
    y: float,
    text: str,
    class_name: str,
    *,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return f"{text[: max_chars - 1]}..."


def r32_card_svg(
    match_id: int,
    x: int,
    y: int,
    height: int,
    slots: pd.DataFrame,
    bracket_row: pd.Series,
    team_emojis: dict[str, str],
) -> str:
    match_slots = slots[slots["match_id"] == match_id]
    elements = [
        (
            f'<rect class="card r32-card" x="{x}" y="{y}" '
            f'width="{CARD_WIDTH}" height="{height}" rx="8" />'
        ),
        svg_text(x + 10, y + 18, f"Match {match_id}", "match-title"),
        svg_text(
            x + CARD_WIDTH - 10,
            y + 18,
            f'to W{int(bracket_row["winner_to"])}',
            "match-meta",
            anchor="end",
        ),
    ]
    current_y = y + 32

    for slot_name in ("home", "away"):
        slot_rows = match_slots[match_slots["slot"] == slot_name]
        if slot_rows.empty:
            raise ValueError(f"No {slot_name} slot rows for match {match_id}")

        source = str(slot_rows.iloc[0]["source"])
        elements.extend(
            (
                f'<line class="slot-rule" x1="{x + 10}" y1="{current_y:.1f}" '
                f'x2="{x + CARD_WIDTH - 10}" y2="{current_y:.1f}" />',
                svg_text(x + 10, current_y + 13, source, "slot-source"),
            )
        )
        current_y += SLOT_BASE_HEIGHT

        for row in slot_rows.sort_values("probability", ascending=False).itertuples():
            elements.append(
                svg_text(
                    x + 10,
                    current_y,
                    truncate_text(format_team_name(row.team, team_emojis), 34),
                    "team-name",
                )
            )
            elements.append(
                svg_text(
                    x + CARD_WIDTH - 10,
                    current_y,
                    probability_label(float(row.probability)),
                    "prob",
                    anchor="end",
                )
            )
            current_y += TEAM_ROW_HEIGHT

    return "\n".join(elements)


def placeholder_card_svg(
    match_id: int,
    x: int,
    y: int,
    height: int,
    bracket: pd.DataFrame,
) -> str:
    bracket_row = bracket[bracket["match_id"] == match_id].iloc[0]
    round_name = str(bracket_row["round"])
    home_source = str(bracket_row["home_source"])
    away_source = str(bracket_row["away_source"])
    title = "Final" if round_name == "F" else round_name
    center_x = x + CARD_WIDTH / 2
    center_y = y + height / 2

    return (
        f'<rect class="card placeholder-card {round_name.lower()}" x="{x}" y="{y}" '
        f'width="{CARD_WIDTH}" height="{height}" rx="8" />\n'
        f'{svg_text(center_x, center_y - 22, title, "round-label", anchor="middle")}\n'
        f'{svg_text(center_x, center_y, f"Match {match_id}", "match-id", anchor="middle")}\n'
        f'{svg_text(center_x, center_y + 22, f"{home_source} v {away_source}", "sources", anchor="middle")}'
    )


def connector_path(
    source: tuple[int, int],
    target: tuple[int, int],
    card_heights: dict[int, int],
    source_match_id: int,
    target_match_id: int,
) -> str:
    source_x, source_y = source
    target_x, target_y = target
    source_center_y = source_y + card_heights[source_match_id] / 2
    target_center_y = target_y + card_heights[target_match_id] / 2

    if source_x < target_x:
        start_x = source_x + CARD_WIDTH
        end_x = target_x
    else:
        start_x = source_x
        end_x = target_x + CARD_WIDTH

    mid_x = (start_x + end_x) / 2
    return (
        f"M {start_x:.1f} {source_center_y:.1f} "
        f"L {mid_x:.1f} {source_center_y:.1f} "
        f"L {mid_x:.1f} {target_center_y:.1f} "
        f"L {end_x:.1f} {target_center_y:.1f}"
    )


def connector_paths(
    bracket: pd.DataFrame,
    positions: dict[int, tuple[int, int]],
    card_heights: dict[int, int],
) -> str:
    paths = []

    for row in bracket.to_dict("records"):
        match_id = int(row["match_id"])
        winner_to = row["winner_to"]
        if pd.isna(winner_to):
            continue

        target_id = int(winner_to)
        if match_id not in positions or target_id not in positions:
            continue

        paths.append(
            '<path class="connector" '
            f'd="{connector_path(positions[match_id], positions[target_id], card_heights, match_id, target_id)}" />'
        )

    return "\n".join(paths)


def embedded_svg_style() -> str:
    return f"""
<style>
  .svg-bg {{
    fill: #ffffff;
  }}

  .connector {{
    fill: none;
    stroke: #65758b;
    stroke-width: 4;
    stroke-linecap: round;
    stroke-linejoin: round;
  }}

  .card {{
    fill: #ffffff;
    stroke: #d4dae4;
    stroke-width: 1;
    filter: drop-shadow(0 1px 1px rgba(23, 32, 51, 0.08));
  }}

  .r32-card {{
    fill: #eef6f4;
  }}

  .placeholder-card {{
    fill: #f3f5f8;
  }}

  .f {{
    fill: #ffffff;
    stroke: #65758b;
    stroke-width: 1.5;
  }}

  text {{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
      sans-serif;
    letter-spacing: 0;
  }}

  .match-title {{
    fill: #172033;
    font-size: 12px;
    font-weight: 760;
  }}

  .match-meta {{
    fill: #667085;
    font-size: 10px;
    font-weight: 700;
  }}

  .slot-rule {{
    stroke: #d5dce7;
    stroke-width: 1;
  }}

  .slot-source {{
    fill: #0f766e;
    font-size: 10px;
    font-weight: 800;
  }}

  .team-name {{
    fill: #172033;
    font-size: 10.5px;
    font-weight: 500;
  }}

  .prob {{
    fill: #172033;
    font-size: 10.5px;
    font-weight: 760;
    font-variant-numeric: tabular-nums;
  }}

  .round-label {{
    fill: #0f766e;
    font-size: 13px;
    font-weight: 820;
    text-transform: uppercase;
  }}

  .match-id {{
    fill: #172033;
    font-size: 17px;
    font-weight: 780;
  }}

  .sources {{
    fill: #667085;
    font-size: 12px;
    font-weight: 650;
  }}

  .eliminated-panel {{
    fill: #f8fafc;
    stroke: #d4dae4;
    stroke-width: 1;
  }}

  .eliminated-title {{
    fill: #991b1b;
    font-size: 13px;
    font-weight: 820;
    text-transform: uppercase;
  }}

  .eliminated-note {{
    fill: #667085;
    font-size: 11px;
    font-weight: 650;
  }}

  .eliminated-chip {{
    fill: #ffffff;
    stroke: #e2e7ef;
    stroke-width: 1;
  }}

  .eliminated-team {{
    fill: #172033;
    font-size: 12px;
    font-weight: 700;
  }}
</style>"""


def eliminated_section_height(eliminated_teams: list[str]) -> int:
    if not eliminated_teams:
        return 0

    rows = max(1, math.ceil(len(eliminated_teams) / ELIMINATED_CHIPS_PER_ROW))
    return ELIMINATED_SECTION_BASE_HEIGHT + rows * ELIMINATED_ROW_HEIGHT


def estimated_text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.58


def eliminated_section_svg(
    x: int,
    y: int,
    height: int,
    eliminated_teams: list[str],
    team_emojis: dict[str, str],
) -> str:
    width = SVG_WIDTH - MARGIN_X * 2
    elements = [
        (
            f'<rect class="eliminated-panel" x="{x}" y="{y}" '
            f'width="{width}" height="{height}" rx="8" />'
        ),
        svg_text(
            x + 16,
            y + 24,
            f"Eliminated ({len(eliminated_teams)})",
            "eliminated-title",
        ),
        svg_text(
            x + width - 16,
            y + 24,
            "No Round of 32 slot paths in the current model output",
            "eliminated-note",
            anchor="end",
        ),
    ]
    chip_x = x + 16
    chip_y = y + 42
    row_start_x = chip_x
    max_x = x + width - 16

    for team in eliminated_teams:
        label = format_team_name(team, team_emojis)
        chip_width = max(92, math.ceil(estimated_text_width(label, 12) + 22))
        if chip_x + chip_width > max_x:
            chip_x = row_start_x
            chip_y += ELIMINATED_ROW_HEIGHT

        elements.append(
            f'<rect class="eliminated-chip" x="{chip_x}" y="{chip_y}" '
            f'width="{chip_width}" height="24" rx="12" />'
        )
        elements.append(
            svg_text(chip_x + 11, chip_y + 16, label, "eliminated-team")
        )
        chip_x += chip_width + 7

    return "\n".join(elements)


def bracket_svg(
    slots: pd.DataFrame,
    bracket: pd.DataFrame,
    team_emojis: dict[str, str],
    eliminated_teams: list[str],
) -> str:
    positions, card_heights, svg_height = match_positions(slots=slots)
    eliminated_height = eliminated_section_height(eliminated_teams)
    export_height = svg_height
    eliminated_section = ""

    if eliminated_teams:
        eliminated_y = svg_height + ELIMINATED_SECTION_TOP_GAP
        export_height += ELIMINATED_SECTION_TOP_GAP + eliminated_height
        eliminated_section = eliminated_section_svg(
            x=MARGIN_X,
            y=eliminated_y,
            height=eliminated_height,
            eliminated_teams=eliminated_teams,
            team_emojis=team_emojis,
        )

    cards = []

    for match_id in LEFT_R32_MATCHES + RIGHT_R32_MATCHES:
        bracket_row = bracket[bracket["match_id"] == match_id].iloc[0]
        x, y = positions[match_id]
        cards.append(
            r32_card_svg(
                match_id=match_id,
                x=x,
                y=y,
                height=card_heights[match_id],
                slots=slots,
                bracket_row=bracket_row,
                team_emojis=team_emojis,
            )
        )

    for match_id in ROUND_LAYOUT:
        x, y = positions[match_id]
        cards.append(
            placeholder_card_svg(
                match_id=match_id,
                x=x,
                y=y,
                height=card_heights[match_id],
                bracket=bracket,
            )
        )

    return (
        f'<svg class="bracket-svg" viewBox="0 0 {SVG_WIDTH} {export_height}" '
        f'width="{SVG_WIDTH}" height="{export_height}" role="img" '
        'aria-label="Round of 32 slot probability bracket">'
        f"{embedded_svg_style()}"
        '<rect class="svg-bg" x="0" y="0" width="100%" height="100%" />'
        '<g class="connectors">'
        f"{connector_paths(bracket=bracket, positions=positions, card_heights=card_heights)}"
        "</g>"
        f"{''.join(cards)}"
        f"{eliminated_section}"
        "</svg>"
    )


def build_html(
    snapshot_date: str,
    slots: pd.DataFrame,
    bracket: pd.DataFrame,
    team_emojis: dict[str, str],
    eliminated_teams: list[str],
) -> str:
    bracket_markup = bracket_svg(
        slots=slots,
        bracket=bracket,
        team_emojis=team_emojis,
        eliminated_teams=eliminated_teams,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Round of 32 Slot Bracket</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #9aa4b2;
      --line-strong: #65758b;
      --accent: #0f766e;
      --r32: #eef6f4;
      --placeholder: #f3f5f8;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
        sans-serif;
    }}

    main {{
      padding: 24px;
    }}

    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin: 0 0 12px;
    }}

    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: 0;
    }}

    .meta {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}

{COPY_CHART_CONTROLS_CSS}

    .bracket-scroll {{
      overflow-x: auto;
      padding-bottom: 12px;
    }}

    .bracket-svg {{
      display: block;
      max-width: none;
      background: #ffffff;
      border: 1px solid #e0e5ee;
      border-radius: 8px;
    }}

    .svg-bg {{
      fill: #ffffff;
    }}

    .connector {{
      fill: none;
      stroke: var(--line-strong);
      stroke-width: 4;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}

    .match-card {{
      width: {CARD_WIDTH}px;
      min-height: {MIN_CARD_HEIGHT}px;
      height: 100%;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid #d4dae4;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(23, 32, 51, 0.06);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
        sans-serif;
    }}

    .match-card.r32 {{
      background: linear-gradient(180deg, var(--panel), var(--r32));
      padding: 8px 9px;
    }}

    .match-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 7px;
      font-size: 12px;
      font-weight: 760;
      line-height: 1.1;
    }}

    .match-head small {{
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
    }}

    .slot {{
      border-top: 1px solid #e2e7ef;
      padding-top: 5px;
    }}

    .slot + .slot {{
      margin-top: 6px;
    }}

    .slot-head {{
      color: var(--accent);
      font-size: 10px;
      font-weight: 800;
      line-height: 1.1;
      margin-bottom: 4px;
    }}

    .team-prob {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: baseline;
      gap: 8px;
      min-height: 15px;
      font-size: 10.5px;
      line-height: 1.16;
    }}

    .team-name {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .prob {{
      color: var(--text);
      font-variant-numeric: tabular-nums;
      font-weight: 760;
      white-space: nowrap;
    }}

    .placeholder {{
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--placeholder);
      border-style: solid;
      padding: 10px;
    }}

    .placeholder-inner {{
      text-align: center;
    }}

    .round-label {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 820;
      margin-bottom: 7px;
      text-transform: uppercase;
    }}

    .match-id {{
      font-size: 17px;
      font-weight: 780;
      margin-bottom: 7px;
    }}

    .sources {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.25;
    }}

    .f {{
      background: #ffffff;
      border-color: var(--line-strong);
      border-width: 1.5px;
    }}

    .note {{
      max-width: 980px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Round of 32 Slot Probability Bracket</h1>
      <div class="meta">Snapshot {escape(snapshot_date)} · all teams shown per slot</div>
    </header>
    <div class="chart-block">
      {copy_chart_button(label="Round of 32 slot bracket", filename="r32-slot-bracket.png")}
      <div class="bracket-scroll">
        {bracket_markup}
      </div>
    </div>
    <p class="note">Only the Round of 32 boxes contain probabilities. Later rounds are structural fixture boxes so the bracket keeps the familiar left and right sides meeting in the middle.</p>
  </main>
  <script>
{COPY_CHART_PNG_SCRIPT}
  </script>
</body>
</html>"""


def build_snapshot_html(
    resources_path: Path,
    snapshot_date_stamp: str,
    team_emojis: dict[str, str],
) -> Path:
    dated_paths = build_dated_resource_paths(
        resources_path=resources_path,
        date_stamp=snapshot_date_stamp,
    )
    slot_path = dated_paths.output_csv_dir / SLOT_CSV_FILE
    if not slot_path.is_file():
        raise FileNotFoundError(
            f"Run calculate-r32-slot-monte-carlo first; missing {slot_path}"
        )

    slots = load_slot_probabilities(slot_path)
    snapshot_teams = load_snapshot_teams(
        dated_paths.input_csv_dir / "group_match_outcome_xg.csv"
    )
    eliminated_teams = sorted(set(snapshot_teams) - set(slots["team"]))
    bracket = load_bracket(resources_path / BRACKET_CSV_FILE)
    html_path = dated_paths.snapshot_dir / "html" / HTML_FILE
    write_text_output(
        html_path,
        build_html(
            snapshot_date=snapshot_date_stamp,
            slots=slots,
            bracket=bracket,
            team_emojis=team_emojis,
            eliminated_teams=eliminated_teams,
        ),
    )

    return html_path


@click.command()
@click.argument(
    "resources_path",
    required=False,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=default_resources_path(),
)
@click.option(
    "--snapshot-date-stamp",
    "--date-stamp",
    "snapshot_date_stamp",
    default=None,
    help=(
        "Dated resource snapshot to read/write, for example 20260624. "
        "Defaults to the latest snapshot containing the R32 slot CSV."
    ),
)
def main(resources_path: Path, snapshot_date_stamp: str | None) -> None:
    resources = load_pool_resources(resources_path)
    slot_path = find_dated_file(
        resources_path=resources_path,
        subdirectory="output_csv",
        filename=SLOT_CSV_FILE,
        date_stamp=snapshot_date_stamp,
    )
    date_stamp = snapshot_date_stamp or slot_path.parent.parent.name
    html_path = build_snapshot_html(
        resources_path=resources_path,
        snapshot_date_stamp=date_stamp,
        team_emojis=resources.team_emojis,
    )
    click.echo(f"Wrote R32 slot bracket HTML to {html_path}")


if __name__ == "__main__":
    main()
