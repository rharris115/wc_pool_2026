from html import escape
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


def slot_rows_html(
    slot_rows: pd.DataFrame,
    team_emojis: dict[str, str],
) -> str:
    slot_rows = slot_rows.sort_values("probability", ascending=False)
    rendered_rows = []

    for row in slot_rows.itertuples():
        rendered_rows.append(
            (
                '<div class="team-prob">'
                f'<span class="team-name">{escape(format_team_name(row.team, team_emojis))}</span>'
                f'<span class="prob">{probability_label(float(row.probability))}</span>'
                "</div>"
            )
        )

    return "\n".join(rendered_rows)


def r32_card_body(
    match_id: int,
    slots: pd.DataFrame,
    bracket_row: pd.Series,
    team_emojis: dict[str, str],
) -> str:
    match_slots = slots[slots["match_id"] == match_id]
    slot_sections = []

    for slot_name in ("home", "away"):
        slot_rows = match_slots[match_slots["slot"] == slot_name]
        if slot_rows.empty:
            raise ValueError(f"No {slot_name} slot rows for match {match_id}")

        source = slot_rows.iloc[0]["source"]
        slot_sections.append(
            (
                '<section class="slot">'
                f'<div class="slot-head"><strong>{escape(str(source))}</strong></div>'
                f"{slot_rows_html(slot_rows=slot_rows, team_emojis=team_emojis)}"
                "</section>"
            )
        )

    return (
        '<div xmlns="http://www.w3.org/1999/xhtml" class="match-card r32">'
        f'<div class="match-head"><span>Match {match_id}</span>'
        f'<small>to W{int(bracket_row["winner_to"])}</small></div>'
        f"{''.join(slot_sections)}"
        "</div>"
    )


def placeholder_body(match_id: int, bracket: pd.DataFrame) -> str:
    bracket_row = bracket[bracket["match_id"] == match_id].iloc[0]
    round_name = str(bracket_row["round"])
    home_source = str(bracket_row["home_source"])
    away_source = str(bracket_row["away_source"])
    title = "Final" if round_name == "F" else round_name

    return (
        '<div xmlns="http://www.w3.org/1999/xhtml" '
        f'class="match-card placeholder {round_name.lower()}">'
        '<div class="placeholder-inner">'
        f'<div class="round-label">{escape(title)}</div>'
        f'<div class="match-id">Match {match_id}</div>'
        f'<div class="sources">{escape(home_source)} v {escape(away_source)}</div>'
        "</div>"
        "</div>"
    )


def card_foreign_object(x: int, y: int, height: int, body: str) -> str:
    return (
        f'<foreignObject x="{x}" y="{y}" width="{CARD_WIDTH}" '
        f'height="{height}">{body}</foreignObject>'
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

  .match-card {{
    width: {CARD_WIDTH}px;
    min-height: {MIN_CARD_HEIGHT}px;
    height: 100%;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid #d4dae4;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(23, 32, 51, 0.06);
    color: #172033;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
      sans-serif;
  }}

  .match-card.r32 {{
    background: linear-gradient(180deg, #ffffff, #eef6f4);
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
    color: #667085;
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
    color: #0f766e;
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
    color: #172033;
    font-variant-numeric: tabular-nums;
    font-weight: 760;
    white-space: nowrap;
  }}

  .other {{
    color: #667085;
  }}

  .placeholder {{
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f3f5f8;
    border-style: solid;
    padding: 10px;
  }}

  .placeholder-inner {{
    text-align: center;
  }}

  .round-label {{
    color: #0f766e;
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
    color: #667085;
    font-size: 12px;
    font-weight: 650;
    line-height: 1.25;
  }}

  .f {{
    background: #ffffff;
    border-color: #65758b;
    border-width: 1.5px;
  }}
</style>"""


def bracket_svg(
    slots: pd.DataFrame,
    bracket: pd.DataFrame,
    team_emojis: dict[str, str],
) -> str:
    positions, card_heights, svg_height = match_positions(slots=slots)
    cards = []

    for match_id in LEFT_R32_MATCHES + RIGHT_R32_MATCHES:
        bracket_row = bracket[bracket["match_id"] == match_id].iloc[0]
        x, y = positions[match_id]
        cards.append(
            card_foreign_object(
                x=x,
                y=y,
                height=card_heights[match_id],
                body=r32_card_body(
                    match_id=match_id,
                    slots=slots,
                    bracket_row=bracket_row,
                    team_emojis=team_emojis,
                ),
            )
        )

    for match_id in ROUND_LAYOUT:
        x, y = positions[match_id]
        cards.append(
            card_foreign_object(
                x=x,
                y=y,
                height=card_heights[match_id],
                body=placeholder_body(match_id=match_id, bracket=bracket),
            )
        )

    return (
        f'<svg class="bracket-svg" viewBox="0 0 {SVG_WIDTH} {svg_height}" '
        f'width="{SVG_WIDTH}" height="{svg_height}" role="img" '
        'aria-label="Round of 32 slot probability bracket">'
        f"{embedded_svg_style()}"
        '<rect class="svg-bg" x="0" y="0" width="100%" height="100%" />'
        '<g class="connectors">'
        f"{connector_paths(bracket=bracket, positions=positions, card_heights=card_heights)}"
        "</g>"
        f"{''.join(cards)}"
        "</svg>"
    )


def build_html(
    snapshot_date: str,
    slots: pd.DataFrame,
    bracket: pd.DataFrame,
    team_emojis: dict[str, str],
) -> str:
    bracket_markup = bracket_svg(
        slots=slots,
        bracket=bracket,
        team_emojis=team_emojis,
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
    bracket = load_bracket(resources_path / BRACKET_CSV_FILE)
    html_path = dated_paths.snapshot_dir / "html" / HTML_FILE
    write_text_output(
        html_path,
        build_html(
            snapshot_date=snapshot_date_stamp,
            slots=slots,
            bracket=bracket,
            team_emojis=team_emojis,
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
