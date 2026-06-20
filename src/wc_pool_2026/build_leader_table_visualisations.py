from html import escape
import math
import re
from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.common import (
    PoolResources,
    dated_snapshot_dirs,
    load_pool_resources,
    require_columns,
)
from wc_pool_2026.paths import default_resources_path
from wc_pool_2026.viz_common import (
    COPY_CHART_CONTROLS_CSS,
    COPY_CHART_PNG_SCRIPT,
    copy_chart_button,
)

ENTRANT_HISTORY_FILE = "leader_table_entrant_history.csv"
TEAM_HISTORY_FILE = "leader_table_team_history.csv"
HTML_FILE = "leader_table_visualisations.html"
ENTROPY_HTML_FILE = "prize_probability_entropy.html"

CHART_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#9333ea",
    "#ca8a04",
    "#0f766e",
    "#475569",
    "#1d4ed8",
    "#b91c1c",
    "#047857",
    "#6d28d9",
    "#c2410c",
    "#0e7490",
    "#9f1239",
    "#3f6212",
]

SVG_CHART_STYLE = """
  .chart-title {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    font-size: 18px;
    font-weight: 700;
    fill: #172033;
  }

  .axis-label {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    font-size: 11px;
    fill: #667085;
  }

  .axis-line {
    stroke: #d7dce5;
    stroke-width: 1.2;
  }

  .grid-line {
    stroke: #edf0f5;
    stroke-width: 1;
  }

  .series-line {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }

  .point-marker {
    font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
      Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 18px;
    font-weight: 700;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 3px;
    stroke-linejoin: round;
  }

  .label-connector {
    fill: none;
    stroke-width: 1.4;
    stroke-linecap: round;
    opacity: 0.72;
  }

  .series-label {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
      sans-serif;
    font-size: 12px;
    font-weight: 700;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 4px;
    stroke-linejoin: round;
  }
"""


def parse_probability_pct(value: str) -> float:
    return float(value.rstrip("%")) / 100


def snapshot_metadata(resources_path: Path) -> list[dict[str, str | int | Path]]:
    snapshots = []

    for match_day, snapshot_dir in enumerate(
        dated_snapshot_dirs(resources_path),
        start=1,
    ):
        snapshots.append(
            {
                "match_day": match_day,
                "snapshot_date": snapshot_dir.name,
                "snapshot_dir": snapshot_dir,
            }
        )

    return snapshots


def load_entrant_history(resources_path: Path) -> pd.DataFrame:
    rows = []

    for snapshot in snapshot_metadata(resources_path):
        output_csv_dir = Path(snapshot["snapshot_dir"]) / "output_csv"
        first_prize_file = output_csv_dir / "first_prize_entrants.csv"
        third_prize_file = output_csv_dir / "third_prize_monte_carlo_entrants.csv"

        if not first_prize_file.is_file() or not third_prize_file.is_file():
            continue

        first_prize = pd.read_csv(first_prize_file)
        third_prize = pd.read_csv(third_prize_file)
        require_columns(
            df=first_prize,
            columns={"rank", "person", "probability_pct", "teams"},
            source=first_prize_file,
        )
        require_columns(
            df=third_prize,
            columns={"rank", "person", "probability_pct", "teams"},
            source=third_prize_file,
        )

        merged = first_prize[["rank", "person", "probability_pct", "teams"]].merge(
            third_prize[["rank", "person", "probability_pct", "teams"]],
            on="person",
            suffixes=("_first_prize", "_third_prize"),
            validate="one_to_one",
        )

        for entrant in merged.to_dict("records"):
            rows.append(
                {
                    "match_day": snapshot["match_day"],
                    "snapshot_date": snapshot["snapshot_date"],
                    "person": entrant["person"],
                    "first_prize_rank": entrant["rank_first_prize"],
                    "first_prize_probability": parse_probability_pct(
                        entrant["probability_pct_first_prize"]
                    ),
                    "first_prize_probability_pct": entrant[
                        "probability_pct_first_prize"
                    ],
                    "first_prize_teams": entrant["teams_first_prize"],
                    "third_prize_rank": entrant["rank_third_prize"],
                    "third_prize_probability": parse_probability_pct(
                        entrant["probability_pct_third_prize"]
                    ),
                    "third_prize_probability_pct": entrant[
                        "probability_pct_third_prize"
                    ],
                    "third_prize_teams": entrant["teams_third_prize"],
                }
            )

    return (
        pd.DataFrame(rows).sort_values(["match_day", "person"]).reset_index(drop=True)
    )


def load_team_history(resources_path: Path, resources: PoolResources) -> pd.DataFrame:
    rows = []

    for snapshot in snapshot_metadata(resources_path):
        output_csv_dir = Path(snapshot["snapshot_dir"]) / "output_csv"
        first_prize_file = output_csv_dir / "first_prize_teams.csv"
        third_prize_file = output_csv_dir / "third_prize_monte_carlo_teams.csv"

        if not first_prize_file.is_file() or not third_prize_file.is_file():
            continue

        first_prize = pd.read_csv(first_prize_file)
        third_prize = pd.read_csv(third_prize_file)
        require_columns(
            df=first_prize,
            columns={"team", "champion_probability"},
            source=first_prize_file,
        )
        require_columns(
            df=third_prize,
            columns={"team", "worst_probability"},
            source=third_prize_file,
        )

        merged = first_prize[["team", "champion_probability"]].merge(
            third_prize[["team", "worst_probability"]],
            on="team",
            validate="one_to_one",
        )

        for team in merged.to_dict("records"):
            champion_probability = team["champion_probability"]
            worst_probability = team["worst_probability"]

            rows.append(
                {
                    "match_day": snapshot["match_day"],
                    "snapshot_date": snapshot["snapshot_date"],
                    "team": team["team"],
                    "team_display": (
                        f"{resources.team_emojis.get(team['team'], '🏳️')} "
                        f"{team['team']}"
                    ),
                    "champion_probability": champion_probability,
                    "champion_probability_pct": (f"{champion_probability * 100:.2f}%"),
                    "worst_team_probability": worst_probability,
                    "worst_team_probability_pct": f"{worst_probability * 100:.2f}%",
                }
            )

    return pd.DataFrame(rows).sort_values(["match_day", "team"]).reset_index(drop=True)


def nice_axis_max(max_value: float) -> float:
    if max_value <= 0.01:
        return 0.01
    if max_value <= 0.05:
        return 0.05
    if max_value <= 0.1:
        return 0.1
    if max_value <= 0.25:
        return 0.25
    if max_value <= 0.5:
        return 0.5
    if max_value <= 0.75:
        return 0.75
    return 1.0


def probability_entropy(probabilities: pd.Series) -> float:
    cleaned = pd.to_numeric(probabilities, errors="coerce").fillna(0)

    if (cleaned < 0).any():
        raise ValueError("Cannot calculate entropy for negative probabilities")

    total = float(cleaned.sum())
    if total <= 0:
        return 0.0

    normalised = cleaned / total

    return -sum(
        probability * math.log2(probability)
        for probability in normalised
        if probability > 0
    )


def build_entropy_history(entrant_history: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (match_day, snapshot_date), group in entrant_history.groupby(
        ["match_day", "snapshot_date"],
        sort=True,
    ):
        first_prize_entropy = probability_entropy(group["first_prize_probability"])
        third_prize_entropy = probability_entropy(group["third_prize_probability"])

        rows.append(
            {
                "match_day": int(match_day),
                "snapshot_date": str(snapshot_date),
                "first_prize_entropy": first_prize_entropy,
                "third_prize_entropy": third_prize_entropy,
                "first_prize_effective_entrants": 2**first_prize_entropy,
                "third_prize_effective_entrants": 2**third_prize_entropy,
            }
        )

    return pd.DataFrame(rows).sort_values("match_day").reset_index(drop=True)


def build_team_color_map(team_history: pd.DataFrame) -> dict[str, str]:
    color_map = {}

    for index, team_display in enumerate(sorted(team_history["team_display"].unique())):
        team = str(team_display).split(" ", maxsplit=1)[-1]
        color_map[team] = CHART_COLORS[index % len(CHART_COLORS)]

    return color_map


def render_colored_team_detail(
    detail: str,
    team_color_map: dict[str, str] | None,
) -> str:
    if not team_color_map:
        return escape(detail)

    matches = list(re.finditer(r"(?:^|\s*\|\s*|\s+)(.+?\([0-9.]+%\))", detail))

    if not matches:
        return escape(detail)

    elements = []
    position = 0
    teams_by_length = sorted(team_color_map, key=len, reverse=True)

    for match in matches:
        chunk = match.group(1)
        team = next((team for team in teams_by_length if team in chunk), None)

        elements.append(escape(detail[position : match.start(1)]))

        if team is None:
            elements.append(escape(chunk))
        else:
            elements.append(
                f'<tspan fill="{team_color_map[team]}">{escape(chunk)}</tspan>'
            )

        position = match.end(1)

    elements.append(escape(detail[position:]))

    return "".join(elements)


def build_svg_line_chart(
    df: pd.DataFrame,
    entity_column: str,
    value_column: str,
    title: str,
    marker_column: str,
    label_column: str | None = None,
    label_detail_column: str | None = None,
    label_detail_color_map: dict[str, str] | None = None,
    show_label_marker: bool = False,
    show_label_rank: bool = True,
    show_label_value: bool = False,
) -> str:
    width = 1800
    height = 940
    margin_left = 62
    margin_right = 1030
    margin_top = 42
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    days = sorted(df["match_day"].unique())
    entities = sorted(df[entity_column].unique())
    max_value = nice_axis_max(float(df[value_column].max()) * 1.08)

    def x_for_day(match_day: int) -> float:
        if len(days) == 1:
            return margin_left + plot_width / 2

        return margin_left + ((match_day - days[0]) / (days[-1] - days[0])) * plot_width

    def y_for_value(value: float) -> float:
        return margin_top + plot_height - (value / max_value) * plot_height

    marker_base_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    arrowhead_ids = {
        color: f"label-arrowhead-{marker_base_id}-{index}"
        for index, color in enumerate(CHART_COLORS)
    }
    marker_defs = ["<defs>"]

    for color, marker_id in arrowhead_ids.items():
        marker_defs.extend(
            [
                f'<marker id="{marker_id}" viewBox="0 0 8 8" refX="7" '
                'refY="4" markerWidth="6" markerHeight="6" orient="auto">',
                f'<path d="M 0 0 L 8 4 L 0 8 z" fill="{color}" />',
                "</marker>",
            ]
        )

    marker_defs.append("</defs>")

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}">',
        f"<style>{SVG_CHART_STYLE}</style>",
        *marker_defs,
        f'<text class="chart-title" x="{margin_left}" y="24">{escape(title)}</text>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        y = y_for_value(value)
        elements.append(
            f'<line class="grid-line" x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" />'
        )
        elements.append(
            f'<text class="axis-label" x="{margin_left - 10}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value * 100:.0f}%</text>'
        )

    for day in days:
        x = x_for_day(day)
        elements.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{margin_top}" '
            f'x2="{x:.1f}" y2="{height - margin_bottom}" />'
        )
        elements.append(
            f'<text class="axis-label" x="{x:.1f}" y="{height - 18}" '
            f'text-anchor="middle">MD{day}</text>'
        )

    elements.append(
        f'<line class="axis-line" x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" />'
    )
    elements.append(
        f'<line class="axis-line" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{height - margin_bottom}" />'
    )

    endpoint_labels = []

    for index, entity in enumerate(entities):
        entity_rows = df[df[entity_column] == entity].sort_values("match_day")
        color = CHART_COLORS[index % len(CHART_COLORS)]
        marker = str(entity_rows[marker_column].iloc[0])
        label = str(
            entity_rows[label_column].iloc[0] if label_column is not None else entity
        )
        label_detail = (
            str(entity_rows[label_detail_column].iloc[-1])
            if label_detail_column is not None
            else None
        )
        points = [
            (
                f"{x_for_day(int(row.match_day)):.1f},"
                f"{y_for_value(float(getattr(row, value_column))):.1f}"
            )
            for row in entity_rows.itertuples()
        ]

        elements.append(
            f'<polyline class="series-line" points="{" ".join(points)}" '
            f'stroke="{color}" />'
        )

        for row in entity_rows.itertuples():
            value = float(getattr(row, value_column))
            elements.append(
                f'<text class="point-marker" '
                f'x="{x_for_day(int(row.match_day)):.1f}" '
                f'y="{y_for_value(value):.1f}" '
                f'fill="{color}" text-anchor="middle" '
                f'dominant-baseline="central">{escape(marker)}'
                f"<title>{escape(str(entity))}: {value * 100:.2f}% "
                f"on MD{int(row.match_day)}</title></text>"
            )

        final_row = entity_rows.iloc[-1]
        endpoint_labels.append(
            {
                "label": label,
                "color": color,
                "marker": marker,
                "point_x": x_for_day(int(final_row["match_day"])),
                "point_y": y_for_value(float(final_row[value_column])),
                "latest_value": float(final_row[value_column]),
                "arrowhead_id": arrowhead_ids[color],
                "label_detail": label_detail,
            }
        )

    medal_prefixes = {1: "🥇", 2: "🥈", 3: "🥉"}
    endpoint_labels.sort(
        key=lambda item: (-item["latest_value"], str(item["label"]).casefold())
    )
    label_x = width - margin_right + 150
    label_top = margin_top + 12
    label_bottom = height - margin_bottom - 12
    label_gap = (
        0
        if len(endpoint_labels) <= 1
        else (label_bottom - label_top) / (len(endpoint_labels) - 1)
    )

    for index, item in enumerate(endpoint_labels):
        item["label_y"] = (
            margin_top + plot_height / 2
            if len(endpoint_labels) <= 1
            else label_top + index * label_gap
        )
        rank = index + 1
        prefix = medal_prefixes.get(rank, f"{rank}.")
        label_text = f"{prefix} {item['label']}" if show_label_rank else item["label"]
        if show_label_marker:
            label_text = f"{prefix} {item['marker']} {item['label']}"
        if show_label_value:
            label_text = f"{label_text} ({item['latest_value'] * 100:.2f}%)"
        label_content = escape(label_text)

        if item["label_detail"]:
            label_content += " → " + render_colored_team_detail(
                detail=str(item["label_detail"]),
                team_color_map=label_detail_color_map,
            )

        elements.append(
            f'<path class="label-connector" '
            f'd="M {label_x - 10:.1f} {item["label_y"]:.1f} '
            f'L {item["point_x"] + 13:.1f} {item["point_y"]:.1f}" '
            f'stroke="{item["color"]}" '
            f'marker-end="url(#{item["arrowhead_id"]})" />'
        )
        elements.append(
            f'<text class="series-label" x="{label_x:.1f}" '
            f'y="{item["label_y"]:.1f}" fill="{item["color"]}" '
            f'dominant-baseline="central">{label_content}</text>'
        )

    elements.append("</svg>")

    return (
        '<div class="chart-block">'
        + copy_chart_button(label=title)
        + f'{"".join(elements)}'
        "</div>"
    )


def build_two_series_svg(
    history: pd.DataFrame,
    title: str,
    y_unit: str,
    first_column: str,
    third_column: str,
    first_label: str,
    third_label: str,
    value_format: str,
    axis_max: float | None = None,
) -> str:
    width = 1180
    height = 620
    margin_left = 72
    margin_right = 170
    margin_top = 48
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    days = sorted(history["match_day"].unique())
    max_value = max(
        float(history[first_column].max()),
        float(history[third_column].max()),
    )
    y_axis_max = axis_max or max(1.0, math.ceil(max_value * 10) / 10)
    series = [
        {
            "label": first_label,
            "column": first_column,
            "color": "#2563eb",
        },
        {
            "label": third_label,
            "column": third_column,
            "color": "#dc2626",
        },
    ]

    def x_for_day(match_day: int) -> float:
        if len(days) == 1:
            return margin_left + plot_width / 2

        return margin_left + ((match_day - days[0]) / (days[-1] - days[0])) * plot_width

    def y_for_value(value: float) -> float:
        return margin_top + plot_height - (value / y_axis_max) * plot_height

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}">',
        f"<style>{SVG_CHART_STYLE}</style>",
        f'<text class="chart-title" x="{margin_left}" y="28">{escape(title)}</text>',
    ]

    for tick in range(6):
        value = y_axis_max * tick / 5
        y = y_for_value(value)
        elements.append(
            f'<line class="grid-line" x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" />'
        )
        elements.append(
            f'<text class="axis-label" x="{margin_left - 10}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value_format.format(value)} {escape(y_unit)}</text>'
        )

    for day in days:
        x = x_for_day(day)
        elements.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{margin_top}" '
            f'x2="{x:.1f}" y2="{height - margin_bottom}" />'
        )
        elements.append(
            f'<text class="axis-label" x="{x:.1f}" y="{height - 22}" '
            f'text-anchor="middle">MD{day}</text>'
        )

    elements.append(
        f'<line class="axis-line" x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom}" />'
    )
    elements.append(
        f'<line class="axis-line" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{height - margin_bottom}" />'
    )

    legend_x = width - margin_right + 24
    legend_y = margin_top + 24

    for index, item in enumerate(series):
        column = str(item["column"])
        color = str(item["color"])
        label = str(item["label"])
        points = [
            (
                f"{x_for_day(int(row.match_day)):.1f},"
                f"{y_for_value(float(getattr(row, column))):.1f}"
            )
            for row in history.itertuples()
        ]
        latest = history.iloc[-1]
        latest_value = float(latest[column])

        elements.append(
            f'<polyline class="series-line" points="{" ".join(points)}" '
            f'stroke="{color}" />'
        )

        for row in history.itertuples():
            value = float(getattr(row, column))
            elements.append(
                f'<circle cx="{x_for_day(int(row.match_day)):.1f}" '
                f'cy="{y_for_value(value):.1f}" r="4.2" fill="{color}">'
                f"<title>{escape(label)}: {value_format.format(value)} "
                f"{escape(y_unit)} on "
                f"MD{int(row.match_day)}</title></circle>"
            )

        item_y = legend_y + index * 46
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{item_y}" '
                f'x2="{legend_x + 24}" y2="{item_y}" stroke="{color}" '
                'stroke-width="3" stroke-linecap="round" />',
                f'<text class="series-label" x="{legend_x + 34}" '
                f'y="{item_y}" fill="{color}" dominant-baseline="central">'
                f'{escape(label)}</text>',
                f'<text class="axis-label" x="{legend_x + 34}" '
                f'y="{item_y + 18}" dominant-baseline="central">'
                f'Latest {value_format.format(latest_value)} '
                f'{escape(y_unit)}</text>',
            ]
        )

    elements.append("</svg>")

    latest = history.iloc[-1]
    stats = (
        '<div class="stats">'
        f'<span>Latest snapshot {escape(str(latest["snapshot_date"]))}</span>'
        f'<span>{escape(first_label)} {value_format.format(float(latest[first_column]))} {escape(y_unit)}</span>'
        f'<span>{escape(third_label)} {value_format.format(float(latest[third_column]))} {escape(y_unit)}</span>'
        "</div>"
    )

    return (
        '<div class="chart-block">'
        + copy_chart_button(label=title)
        + "".join(elements)
        + stats
        + "</div>"
    )


def build_entropy_svg(entropy_history: pd.DataFrame) -> str:
    return build_two_series_svg(
        history=entropy_history,
        title="Prize Probability Entropy",
        y_unit="bits",
        first_column="first_prize_entropy",
        third_column="third_prize_entropy",
        first_label="1st Prize",
        third_label="3rd Prize",
        value_format="{:.2f}",
    )


def build_effective_entrants_svg(entropy_history: pd.DataFrame) -> str:
    return build_two_series_svg(
        history=entropy_history,
        title="Effective Entrants",
        y_unit="entrants",
        first_column="first_prize_effective_entrants",
        third_column="third_prize_effective_entrants",
        first_label="1st Prize",
        third_label="3rd Prize",
        value_format="{:.1f}",
    )


def build_entropy_table(entropy_history: pd.DataFrame) -> str:
    rows = []

    for row in entropy_history.itertuples():
        rows.append(
            "<tr>"
            f"<td>MD{int(row.match_day)}</td>"
            f"<td>{escape(str(row.snapshot_date))}</td>"
            f"<td>{float(row.first_prize_entropy):.3f}</td>"
            f"<td>{float(row.third_prize_entropy):.3f}</td>"
            f"<td>{float(row.first_prize_effective_entrants):.1f}</td>"
            f"<td>{float(row.third_prize_effective_entrants):.1f}</td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap">'
        "<table>"
        "<thead>"
        "<tr>"
        "<th>Match Day</th>"
        "<th>Snapshot</th>"
        "<th>1st Prize Entropy</th>"
        "<th>3rd Prize Entropy</th>"
        "<th>1st Effective Entrants</th>"
        "<th>3rd Effective Entrants</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def build_entropy_html(entropy_history: pd.DataFrame) -> str:
    latest_day = int(entropy_history["match_day"].max())
    latest_snapshot = entropy_history[entropy_history["match_day"] == latest_day][
        "snapshot_date"
    ].iloc[0]
    entropy_chart = build_entropy_svg(entropy_history)
    effective_entrants_chart = build_effective_entrants_svg(entropy_history)
    entropy_table = build_entropy_table(entropy_history)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prize Probability Entropy</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d7dce5;
      --grid: #edf0f5;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}

    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}

    header {{
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 720;
    }}

    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
      white-space: nowrap;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
    }}

    .chart-block {{
      overflow-x: auto;
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      margin-top: 8px;
    }}

    .table-wrap {{
      margin-top: 14px;
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    th,
    td {{
      border-top: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}

    th {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }}

    th:first-child,
    td:first-child,
    th:nth-child(2),
    td:nth-child(2) {{
      text-align: left;
    }}

{COPY_CHART_CONTROLS_CSS}

    svg {{
      display: block;
      width: 100%;
      min-width: 900px;
      height: auto;
    }}

    .chart-title {{
      font-size: 18px;
      font-weight: 700;
      fill: var(--text);
    }}

    .axis-label {{
      font-size: 11px;
      fill: var(--muted);
    }}

    .axis-line {{
      stroke: var(--line);
      stroke-width: 1.2;
    }}

    .grid-line {{
      stroke: var(--grid);
      stroke-width: 1;
    }}

    .series-line {{
      fill: none;
      stroke-width: 2.6;
      stroke-linejoin: round;
      stroke-linecap: round;
    }}

    .series-label {{
      font-size: 12px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 4px;
      stroke-linejoin: round;
    }}

    @media (max-width: 720px) {{
      main {{
        padding: 18px 12px 28px;
      }}
      .meta {{
        white-space: normal;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Prize Probability Entropy</h1>
      <div class="meta">Match days 1-{latest_day} · latest snapshot {escape(latest_snapshot)}</div>
    </header>
    <section class="panel">{entropy_chart}</section>
    <section class="panel">{effective_entrants_chart}{entropy_table}</section>
  </main>
  <script>
{COPY_CHART_PNG_SCRIPT}
  </script>
</body>
</html>
"""


def build_html(
    entrant_history: pd.DataFrame,
    team_history: pd.DataFrame,
    resources: PoolResources,
) -> str:
    entrant_history = entrant_history.assign(
        marker=entrant_history["person"].map(
            lambda person: resources.entrant_abbreviations.get(person, person)
        )
    )
    team_history = team_history.assign(
        marker=team_history["team_display"].str.split(" ", n=1).str[0]
    )
    team_color_map = build_team_color_map(team_history)
    latest_day = int(entrant_history["match_day"].max())
    latest_snapshot = entrant_history[entrant_history["match_day"] == latest_day][
        "snapshot_date"
    ].iloc[0]

    entrant_first = build_svg_line_chart(
        df=entrant_history,
        entity_column="person",
        value_column="first_prize_probability",
        title="Entrant 1st Prize Probability",
        marker_column="marker",
        label_detail_column="first_prize_teams",
        label_detail_color_map=team_color_map,
        show_label_marker=True,
        show_label_value=True,
    )
    entrant_third = build_svg_line_chart(
        df=entrant_history,
        entity_column="person",
        value_column="third_prize_probability",
        title="Entrant 3rd Prize Probability",
        marker_column="marker",
        label_detail_column="third_prize_teams",
        label_detail_color_map=team_color_map,
        show_label_marker=True,
        show_label_value=True,
    )
    team_champion = build_svg_line_chart(
        df=team_history,
        entity_column="team_display",
        value_column="champion_probability",
        title="Team Champion Probability",
        marker_column="marker",
        label_column="team",
        show_label_rank=False,
    )
    team_worst = build_svg_line_chart(
        df=team_history,
        entity_column="team_display",
        value_column="worst_team_probability",
        title="Team Worst Group-Stage Probability",
        marker_column="marker",
        label_column="team",
        show_label_rank=False,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leader Table Visualisations</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d7dce5;
      --grid: #edf0f5;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}

    main {{
      max-width: 1450px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}

    header {{
      margin-bottom: 18px;
    }}

    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 720;
    }}

    .meta {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}

    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
    }}

    .chart-block {{
      overflow-x: auto;
    }}

{COPY_CHART_CONTROLS_CSS}

    svg {{
      display: block;
      width: 100%;
      min-width: 1600px;
      height: auto;
    }}

    .chart-title {{
      font-size: 18px;
      font-weight: 700;
      fill: var(--text);
    }}

    .axis-label {{
      font-size: 11px;
      fill: var(--muted);
    }}

    .axis-line {{
      stroke: var(--line);
      stroke-width: 1.2;
    }}

    .grid-line {{
      stroke: var(--grid);
      stroke-width: 1;
    }}

    .series-line {{
      fill: none;
      stroke-width: 2;
      stroke-linejoin: round;
      stroke-linecap: round;
    }}

    .point-marker {{
      font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
        Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 18px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}

    .label-connector {{
      fill: none;
      stroke-width: 1.4;
      stroke-linecap: round;
      opacity: 0.72;
    }}

    .series-label {{
      font-size: 12px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 4px;
      stroke-linejoin: round;
    }}

    @media (max-width: 720px) {{
      main {{
        padding: 18px 12px 28px;
      }}
      .meta {{
        margin-top: 6px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Leader Table Visualisations</h1>
      <div class="meta">Match days 1-{latest_day} · latest snapshot {escape(latest_snapshot)}</div>
    </header>
    <section class="grid">
      <div class="panel">{entrant_first}</div>
      <div class="panel">{entrant_third}</div>
      <div class="panel">{team_champion}</div>
      <div class="panel">{team_worst}</div>
    </section>
  </main>
  <script>
{COPY_CHART_PNG_SCRIPT}
  </script>
</body>
</html>
"""


def build_leader_table_visualisations(
    resources_path: Path,
) -> tuple[Path, Path, Path, Path]:
    resources = load_pool_resources(resources_path)
    entrant_history = load_entrant_history(resources_path)
    team_history = load_team_history(resources_path, resources)

    if entrant_history.empty:
        raise ValueError("No entrant history rows found")

    if team_history.empty:
        raise ValueError("No team history rows found")

    entropy_history = build_entropy_history(entrant_history)
    output_csv_dir = resources_path / "output_csv"
    output_html_dir = resources_path / "output_html"
    entrant_history_path = output_csv_dir / ENTRANT_HISTORY_FILE
    team_history_path = output_csv_dir / TEAM_HISTORY_FILE
    html_path = output_html_dir / HTML_FILE
    entropy_html_path = output_html_dir / ENTROPY_HTML_FILE

    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_html_dir.mkdir(parents=True, exist_ok=True)

    entrant_history.to_csv(entrant_history_path, index=False)
    team_history.to_csv(team_history_path, index=False)
    html_path.write_text(
        build_html(
            entrant_history=entrant_history,
            team_history=team_history,
            resources=resources,
        ),
        encoding="utf-8",
    )
    entropy_html_path.write_text(
        build_entropy_html(entropy_history=entropy_history),
        encoding="utf-8",
    )

    return entrant_history_path, team_history_path, html_path, entropy_html_path


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
    entrant_history_path, team_history_path, html_path, entropy_html_path = (
        build_leader_table_visualisations(resources_path)
    )

    click.echo(f"Wrote entrant history CSV to {entrant_history_path}")
    click.echo(f"Wrote team history CSV to {team_history_path}")
    click.echo(f"Wrote leader table visualisations HTML to {html_path}")
    click.echo(f"Wrote prize probability entropy HTML to {entropy_html_path}")


if __name__ == "__main__":
    main()
