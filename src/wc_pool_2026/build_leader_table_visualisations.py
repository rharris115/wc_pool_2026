from html import escape
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

ENTRANT_HISTORY_FILE = "leader_table_entrant_history.csv"
TEAM_HISTORY_FILE = "leader_table_team_history.csv"
HTML_FILE = "leader_table_visualisations.html"

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

ENTRANT_MARKERS = [
    "◆",
    "●",
    "▲",
    "■",
    "✦",
    "✚",
    "✕",
    "✹",
    "◇",
    "○",
    "△",
    "□",
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
            columns={"person", "probability_pct"},
            source=first_prize_file,
        )
        require_columns(
            df=third_prize,
            columns={"person", "probability_pct"},
            source=third_prize_file,
        )

        merged = first_prize[["person", "probability_pct"]].merge(
            third_prize[["person", "probability_pct"]],
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
                    "first_prize_probability": parse_probability_pct(
                        entrant["probability_pct_first_prize"]
                    ),
                    "first_prize_probability_pct": entrant[
                        "probability_pct_first_prize"
                    ],
                    "third_prize_probability": parse_probability_pct(
                        entrant["probability_pct_third_prize"]
                    ),
                    "third_prize_probability_pct": entrant[
                        "probability_pct_third_prize"
                    ],
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


def build_svg_line_chart(
    df: pd.DataFrame,
    entity_column: str,
    value_column: str,
    title: str,
    marker_column: str,
    label_column: str | None = None,
    show_label_rank: bool = True,
) -> str:
    width = 1160
    height = 940
    margin_left = 62
    margin_right = 390
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
                "point_x": x_for_day(int(final_row["match_day"])),
                "point_y": y_for_value(float(final_row[value_column])),
                "latest_value": float(final_row[value_column]),
                "arrowhead_id": arrowhead_ids[color],
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
            f'dominant-baseline="central">{escape(label_text)}</text>'
        )

    elements.append("</svg>")

    return (
        '<div class="chart-block">'
        '<div class="chart-actions">'
        f'<button class="copy-chart" type="button" aria-label="Copy {escape(title)} as PNG">'
        "Copy PNG</button>"
        '<span class="copy-status" aria-live="polite"></span>'
        "</div>"
        f'{"".join(elements)}'
        "</div>"
    )


def build_html(
    entrant_history: pd.DataFrame,
    team_history: pd.DataFrame,
) -> str:
    people = sorted(entrant_history["person"].unique())
    entrant_marker_map = {
        person: ENTRANT_MARKERS[index % len(ENTRANT_MARKERS)]
        for index, person in enumerate(people)
    }
    entrant_history = entrant_history.assign(
        marker=entrant_history["person"].map(entrant_marker_map)
    )
    team_history = team_history.assign(
        marker=team_history["team_display"].str.split(" ", n=1).str[0]
    )
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
    )
    entrant_third = build_svg_line_chart(
        df=entrant_history,
        entity_column="person",
        value_column="third_prize_probability",
        title="Entrant 3rd Prize Probability",
        marker_column="marker",
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
      max-width: 980px;
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

    .chart-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }}

    .copy-chart {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      line-height: 1;
      padding: 8px 10px;
    }}

    .copy-chart:hover {{
      border-color: #aeb7c6;
      background: #f9fafb;
    }}

    .copy-chart:disabled {{
      cursor: progress;
      opacity: 0.64;
    }}

    .copy-status {{
      color: var(--muted);
      font-size: 12px;
      min-height: 1em;
    }}

    svg {{
      display: block;
      width: 100%;
      min-width: 760px;
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
    async function svgToPngBlob(svg) {{
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

      const viewBox = clone.viewBox.baseVal;
      const width = viewBox && viewBox.width ? viewBox.width : clone.clientWidth;
      const height = viewBox && viewBox.height ? viewBox.height : clone.clientHeight;
      const scale = Math.max(2, window.devicePixelRatio || 1);
      clone.setAttribute("width", width);
      clone.setAttribute("height", height);
      clone.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const background = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "rect"
      );
      background.setAttribute("x", "0");
      background.setAttribute("y", "0");
      background.setAttribute("width", width);
      background.setAttribute("height", height);
      background.setAttribute("fill", "#ffffff");
      clone.insertBefore(background, clone.firstChild);

      const serializer = new XMLSerializer();
      const svgText = serializer.serializeToString(clone);
      const svgBlob = new Blob([svgText], {{
        type: "image/svg+xml;charset=utf-8",
      }});
      const url = URL.createObjectURL(svgBlob);

      try {{
        const image = new Image();
        image.decoding = "async";
        image.src = url;
        await image.decode();

        const canvas = document.createElement("canvas");
        canvas.width = Math.ceil(width * scale);
        canvas.height = Math.ceil(height * scale);

        const context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);

        return await new Promise((resolve, reject) => {{
          canvas.toBlob((blob) => {{
            if (blob) {{
              resolve(blob);
            }} else {{
              reject(new Error("Could not create PNG"));
            }}
          }}, "image/png");
        }});
      }} finally {{
        URL.revokeObjectURL(url);
      }}
    }}

    async function copyChartAsPng(button) {{
      const block = button.closest(".chart-block");
      const svg = block.querySelector("svg");
      const status = block.querySelector(".copy-status");

      if (!navigator.clipboard || !window.ClipboardItem) {{
        status.textContent = "PNG clipboard copy is not supported in this browser.";
        return;
      }}

      button.disabled = true;
      status.textContent = "Copying...";

      try {{
        const blob = await svgToPngBlob(svg);
        await navigator.clipboard.write([
          new ClipboardItem({{
            [blob.type]: blob,
          }}),
        ]);
        status.textContent = "Copied PNG";
      }} catch (error) {{
        console.error(error);
        status.textContent = "Copy failed. Try opening this page in Chrome or Safari.";
      }} finally {{
        button.disabled = false;
        window.setTimeout(() => {{
          if (status.textContent === "Copied PNG") {{
            status.textContent = "";
          }}
        }}, 2200);
      }}
    }}

    document.querySelectorAll(".copy-chart").forEach((button) => {{
      button.addEventListener("click", () => copyChartAsPng(button));
    }});
  </script>
</body>
</html>
"""


def build_leader_table_visualisations(resources_path: Path) -> tuple[Path, Path, Path]:
    resources = load_pool_resources(resources_path)
    entrant_history = load_entrant_history(resources_path)
    team_history = load_team_history(resources_path, resources)

    if entrant_history.empty:
        raise ValueError("No entrant history rows found")

    if team_history.empty:
        raise ValueError("No team history rows found")

    output_csv_dir = resources_path / "output_csv"
    output_html_dir = resources_path / "output_html"
    entrant_history_path = output_csv_dir / ENTRANT_HISTORY_FILE
    team_history_path = output_csv_dir / TEAM_HISTORY_FILE
    html_path = output_html_dir / HTML_FILE

    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_html_dir.mkdir(parents=True, exist_ok=True)

    entrant_history.to_csv(entrant_history_path, index=False)
    team_history.to_csv(team_history_path, index=False)
    html_path.write_text(
        build_html(
            entrant_history=entrant_history,
            team_history=team_history,
        ),
        encoding="utf-8",
    )

    return entrant_history_path, team_history_path, html_path


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
    entrant_history_path, team_history_path, html_path = (
        build_leader_table_visualisations(resources_path)
    )

    click.echo(f"Wrote entrant history CSV to {entrant_history_path}")
    click.echo(f"Wrote team history CSV to {team_history_path}")
    click.echo(f"Wrote leader table visualisations HTML to {html_path}")


if __name__ == "__main__":
    main()
