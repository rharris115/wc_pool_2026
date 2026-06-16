from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

import click
import pandas as pd

from wc_pool_2026.common import (
    MAX_GOALS_FOR_OUTCOME_PROBS,
    dated_snapshot_dirs,
    load_pool_resources,
    poisson_probs,
    require_columns,
    write_text_output,
)
from wc_pool_2026.paths import default_resources_path
from wc_pool_2026.viz_common import (
    COPY_CHART_CONTROLS_CSS,
    COPY_CHART_PNG_SCRIPT,
    copy_chart_button,
    filename_slug,
)

HTML_FILE = "goal_difference_histograms.html"
PROBABILITY_THRESHOLD_FOR_AXIS = 0.0005
SVG_CHART_STYLE = """
  .match-title {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    font-size: 13px;
    font-weight: 760;
    fill: #172033;
  }

  .side-label {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
      sans-serif;
    fill: #172033;
    font-size: 11px;
    font-weight: 760;
  }

  .axis-label {
    fill: #667085;
    font-size: 9px;
    font-weight: 650;
  }

  .grid-line {
    stroke: #edf0f5;
    stroke-width: 1;
  }

  .axis-line {
    stroke: #d7dce5;
    stroke-width: 1.2;
  }

  .expected-line {
    stroke: #172033;
    stroke-width: 1.4;
    stroke-dasharray: 4 3;
  }

  .negative-bar {
    fill: #dc2626;
    opacity: 0.78;
  }

  .zero-bar {
    fill: #667085;
    opacity: 0.72;
  }

  .positive-bar {
    fill: #059669;
    opacity: 0.78;
  }
"""


@dataclass(frozen=True)
class MatchGoalDifference:
    match_id: str
    commence_time: str
    plot_index: int
    team: str
    opponent: str
    team_xg: float
    opponent_xg: float
    distribution: dict[int, float]


def match_goal_difference_distribution(
    team_xg: float,
    opponent_xg: float,
    max_goals: int = MAX_GOALS_FOR_OUTCOME_PROBS,
) -> dict[int, float]:
    team_goal_probs = poisson_probs(team_xg, max_goals)
    opponent_goal_probs = poisson_probs(opponent_xg, max_goals)
    distribution: dict[int, float] = defaultdict(float)

    for team_goals, team_prob in enumerate(team_goal_probs):
        for opponent_goals, opponent_prob in enumerate(opponent_goal_probs):
            distribution[team_goals - opponent_goals] += team_prob * opponent_prob

    return normalise_distribution(dict(distribution))


def normalise_distribution(distribution: dict[int, float]) -> dict[int, float]:
    total = sum(distribution.values())

    if total <= 0:
        raise ValueError("Cannot normalise an empty goal-difference distribution")

    return {
        goal_difference: probability / total
        for goal_difference, probability in distribution.items()
    }


def match_goal_difference_rows(match_xg: pd.DataFrame) -> list[MatchGoalDifference]:
    match_rows = match_xg.sort_values(["commence_time", "match_id"]).reset_index(
        drop=True
    )

    return [
        MatchGoalDifference(
            match_id=str(row.match_id),
            commence_time=str(row.commence_time),
            plot_index=index,
            team=str(row.team),
            opponent=str(row.opponent),
            team_xg=float(row.team_xg),
            opponent_xg=float(row.opponent_xg),
            distribution=match_goal_difference_distribution(
                team_xg=float(row.team_xg),
                opponent_xg=float(row.opponent_xg),
            ),
        )
        for index, row in enumerate(match_rows.itertuples())
    ]


def match_goal_difference_pairs(
    matches: list[MatchGoalDifference],
) -> list[list[MatchGoalDifference]]:
    pairs_by_match_id: dict[str, list[MatchGoalDifference]] = defaultdict(list)

    for match in matches:
        pairs_by_match_id[match.match_id].append(match)

    return sorted(
        (
            sorted(pair, key=lambda match: match.plot_index)
            for pair in pairs_by_match_id.values()
        ),
        key=lambda pair: (pair[0].commence_time, pair[0].plot_index),
    )


def expected_goal_difference(distribution: dict[int, float]) -> float:
    return sum(
        goal_difference * probability
        for goal_difference, probability in distribution.items()
    )


def probability_less_than_zero(distribution: dict[int, float]) -> float:
    return sum(
        probability
        for goal_difference, probability in distribution.items()
        if goal_difference < 0
    )


def probability_greater_than_zero(distribution: dict[int, float]) -> float:
    return sum(
        probability
        for goal_difference, probability in distribution.items()
        if goal_difference > 0
    )


def display_goal_difference_range(
    matches: list[MatchGoalDifference],
) -> range:
    included_goal_differences = [
        goal_difference
        for match in matches
        for goal_difference, probability in match.distribution.items()
        if probability >= PROBABILITY_THRESHOLD_FOR_AXIS
    ]

    if not included_goal_differences:
        return range(-1, 2)

    minimum = min(min(included_goal_differences), -1)
    maximum = max(max(included_goal_differences), 1)

    return range(minimum, maximum + 1)


def display_match_time(commence_time: str) -> str:
    timestamp = pd.Timestamp(commence_time)
    return timestamp.strftime("%d %b %Y %H:%M UTC")


def build_match_histogram_svg(
    match: MatchGoalDifference,
    team_emoji: str,
    opponent_emoji: str,
    goal_difference_range: range,
) -> str:
    width = 460
    height = 198
    margin_left = 40
    margin_right = 18
    margin_top = 42
    margin_bottom = 26
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    goal_differences = list(goal_difference_range)
    max_probability = max(
        match.distribution.get(goal_difference, 0.0)
        for goal_difference in goal_differences
    )
    max_probability = max(max_probability, 0.01)
    bar_gap = 1.2
    bar_width = max(
        1.5,
        (plot_width - bar_gap * (len(goal_differences) - 1)) / len(goal_differences),
    )

    def x_for_index(index: int) -> float:
        return margin_left + index * (bar_width + bar_gap)

    def y_for_probability(probability: float) -> float:
        return margin_top + plot_height - (probability / max_probability) * plot_height

    expected = expected_goal_difference(match.distribution)
    negative_label = f"{opponent_emoji} {match.opponent}"
    positive_label = f"{team_emoji} {match.team}"
    elements = [
        f'<svg class="histogram" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(match.team)} versus {escape(match.opponent)} '
        'goal difference histogram">',
        f"<style>{SVG_CHART_STYLE}</style>",
        f'<text class="match-title" x="{width / 2:.1f}" y="17" '
        f'text-anchor="middle">'
        f"{escape(display_match_time(match.commence_time))}</text>",
        f'<text class="side-label" x="{margin_left}" y="34" text-anchor="start">'
        f"{escape(negative_label)}</text>",
        f'<text class="side-label" x="{width - margin_right}" y="34" '
        f'text-anchor="end">{escape(positive_label)}</text>',
    ]

    for tick in range(3):
        probability = max_probability * tick / 2
        y = y_for_probability(probability)
        elements.append(
            f'<line class="grid-line" x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" />'
        )
        elements.append(
            f'<text class="axis-label" x="{margin_left - 6}" y="{y + 3:.1f}" '
            f'text-anchor="end">{probability * 100:.0f}%</text>'
        )

    if goal_differences[0] <= expected <= goal_differences[-1]:
        expected_x = (
            margin_left
            + (expected - goal_differences[0]) * (bar_width + bar_gap)
            + bar_width / 2
        )
        elements.append(
            f'<line class="expected-line" x1="{expected_x:.1f}" '
            f'y1="{margin_top}" x2="{expected_x:.1f}" '
            f'y2="{height - margin_bottom}">'
            f"<title>Mean GD {expected:+.2f}</title></line>"
        )

    for index, goal_difference in enumerate(goal_differences):
        probability = match.distribution.get(goal_difference, 0.0)
        x = x_for_index(index)
        y = y_for_probability(probability)
        bar_height = height - margin_bottom - y
        bar_class = (
            "positive-bar"
            if goal_difference > 0
            else "negative-bar" if goal_difference < 0 else "zero-bar"
        )
        elements.append(
            f'<rect class="{bar_class}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" height="{bar_height:.1f}">'
            f"<title>GD {goal_difference:+d}: {probability * 100:.2f}%</title>"
            "</rect>"
        )

    x_axis_y = height - margin_bottom
    elements.append(
        f'<line class="axis-line" x1="{margin_left}" y1="{x_axis_y}" '
        f'x2="{width - margin_right}" y2="{x_axis_y}" />'
    )
    for goal_difference in (goal_differences[0], 0, goal_differences[-1]):
        if goal_difference not in goal_differences:
            continue

        index = goal_differences.index(goal_difference)
        x = x_for_index(index) + bar_width / 2
        elements.append(
            f'<text class="axis-label" x="{x:.1f}" y="{height - 10}" '
            f'text-anchor="middle">{goal_difference:+d}</text>'
        )

    elements.append("</svg>")

    opponent_win_probability = probability_less_than_zero(match.distribution)
    team_win_probability = probability_greater_than_zero(match.distribution)
    draw_probability = match.distribution.get(0, 0.0)

    return (
        '<article class="card chart-block">'
        + copy_chart_button(
            label=f"{match.team} versus {match.opponent}",
            filename=(
                f"{filename_slug(match.commence_time)}-"
                f"{filename_slug(match.team)}-vs-"
                f"{filename_slug(match.opponent)}.png"
            ),
        )
        + f'{"".join(elements)}'
        '<div class="stats">'
        f"<span>xG {match.team_xg:.2f}-{match.opponent_xg:.2f}</span>"
        f"<span>Mean GD {expected:+.2f}</span>"
        f"<span>{escape(match.team)} {team_win_probability * 100:.0f}%</span>"
        f"<span>Draw {draw_probability * 100:.0f}%</span>"
        f"<span>{escape(match.opponent)} {opponent_win_probability * 100:.0f}%</span>"
        "</div>"
        "</article>"
    )


def build_html(
    snapshot_date: str,
    match_xg: pd.DataFrame,
    team_emojis: dict[str, str],
) -> str:
    matches = match_goal_difference_rows(match_xg)
    match_pairs = match_goal_difference_pairs(matches)
    goal_difference_range = display_goal_difference_range(matches)
    match_rows = "\n".join(
        (
            '<section class="match-row">'
            + "".join(
                build_match_histogram_svg(
                    match=match,
                    team_emoji=team_emojis.get(match.team, "🏳️"),
                    opponent_emoji=team_emojis.get(match.opponent, "🏳️"),
                    goal_difference_range=goal_difference_range,
                )
                for match in pair
            )
            + "</section>"
        )
        for pair in match_pairs
    )
    min_goal_difference = goal_difference_range.start
    max_goal_difference = goal_difference_range.stop - 1

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Match Goal Difference Histograms {escape(snapshot_date)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d7dce5;
      --grid: #edf0f5;
      --negative: #dc2626;
      --zero: #667085;
      --positive: #059669;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}

    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}

    header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }}

    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 720;
    }}

    .meta {{
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      white-space: nowrap;
    }}

    .match-list {{
      display: grid;
      gap: 12px;
    }}

    .match-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    .card {{
      background: var(--panel);
      border: 1px solid #e3e7ee;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(23, 32, 51, 0.05);
      padding: 10px 10px 8px;
    }}

{COPY_CHART_CONTROLS_CSS}

    .histogram {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .match-title {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
        sans-serif;
      font-size: 13px;
      font-weight: 760;
    }}

    .side-label {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
        sans-serif;
      fill: var(--text);
      font-size: 11px;
      font-weight: 760;
    }}

    .axis-label {{
      fill: #667085;
      font-size: 9px;
      font-weight: 650;
    }}

    .grid-line {{
      stroke: var(--grid);
      stroke-width: 1;
    }}

    .axis-line {{
      stroke: var(--line);
      stroke-width: 1.2;
    }}

    .negative-bar {{
      fill: var(--negative);
      opacity: 0.78;
    }}

    .zero-bar {{
      fill: var(--zero);
      opacity: 0.72;
    }}

    .positive-bar {{
      fill: var(--positive);
      opacity: 0.78;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
    }}

    .stats span {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    @media (max-width: 720px) {{
      main {{
        padding: 20px 10px 28px;
      }}

      header {{
        display: block;
      }}

      .meta {{
        margin-top: 4px;
        text-align: left;
        white-space: normal;
      }}

      .match-row {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Match Goal Difference Histograms</h1>
      <div class="meta">Snapshot {escape(snapshot_date)} · {len(match_pairs)} matches · {len(matches)} plots · GD axis {min_goal_difference:+d} to {max_goal_difference:+d}</div>
    </header>
    <div class="match-list">
      {match_rows}
    </div>
  </main>
  <script>
{COPY_CHART_PNG_SCRIPT}
  </script>
</body>
</html>"""


def build_snapshot_html(snapshot_dir: Path, team_emojis: dict[str, str]) -> Path | None:
    match_xg_path = snapshot_dir / "input_csv" / "group_match_outcome_xg.csv"

    if not match_xg_path.is_file():
        return None

    match_xg = pd.read_csv(match_xg_path)
    require_columns(
        df=match_xg,
        columns={
            "match_id",
            "commence_time",
            "team",
            "opponent",
            "team_xg",
            "opponent_xg",
        },
        source=match_xg_path,
    )

    html_path = snapshot_dir / "html" / HTML_FILE
    write_text_output(
        html_path,
        build_html(
            snapshot_date=snapshot_dir.name,
            match_xg=match_xg,
            team_emojis=team_emojis,
        ),
    )

    return html_path


def build_all_snapshot_html(resources_path: Path) -> list[Path]:
    resources = load_pool_resources(resources_path)
    html_paths = []

    for snapshot_dir in dated_snapshot_dirs(resources_path):
        html_path = build_snapshot_html(
            snapshot_dir=snapshot_dir,
            team_emojis=resources.team_emojis,
        )

        if html_path is not None:
            html_paths.append(html_path)

    return html_paths


@click.command()
@click.argument(
    "resources_path",
    required=False,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=default_resources_path(),
)
def main(resources_path: Path) -> None:
    html_paths = build_all_snapshot_html(resources_path=resources_path)

    if not html_paths:
        raise click.ClickException(
            f"No dated snapshots with input_csv/group_match_outcome_xg.csv found in {resources_path}"
        )

    for html_path in html_paths:
        click.echo(f"Wrote goal difference histograms HTML to {html_path}")


if __name__ == "__main__":
    main()
