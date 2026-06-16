from html import escape
import re


COPY_CHART_CONTROLS_CSS = """
    .chart-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }

    .copy-chart {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      line-height: 1;
      padding: 7px 9px;
    }

    .copy-chart:hover {
      border-color: #aeb7c6;
      background: #f9fafb;
    }

    .copy-chart:disabled {
      cursor: progress;
      opacity: 0.64;
    }

    .copy-status {
      color: var(--muted);
      font-size: 12px;
      min-height: 1em;
    }
"""

COPY_CHART_PNG_SCRIPT = """
    async function svgToPngBlob(svg, stats) {
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

      const viewBox = clone.viewBox.baseVal;
      const width = viewBox && viewBox.width ? viewBox.width : clone.clientWidth;
      const height = viewBox && viewBox.height ? viewBox.height : clone.clientHeight;
      const statTexts = stats
        ? Array.from(stats.querySelectorAll("span")).map((span) =>
            span.textContent.trim()
          )
        : [];
      const statsHeight = statTexts.length ? 32 : 0;
      const exportHeight = height + statsHeight;
      const scale = Math.max(2, window.devicePixelRatio || 1);
      clone.setAttribute("width", width);
      clone.setAttribute("height", exportHeight);
      clone.setAttribute("viewBox", `0 0 ${width} ${exportHeight}`);

      const background = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "rect"
      );
      background.setAttribute("x", "0");
      background.setAttribute("y", "0");
      background.setAttribute("width", width);
      background.setAttribute("height", exportHeight);
      background.setAttribute("fill", "#ffffff");
      clone.insertBefore(background, clone.firstChild);

      if (statTexts.length) {
        const statsGroup = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "g"
        );
        const columnWidth = width / statTexts.length;

        statTexts.forEach((statText, index) => {
          const text = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "text"
          );
          text.setAttribute("x", columnWidth * index + columnWidth / 2);
          text.setAttribute("y", height + 21);
          text.setAttribute("text-anchor", "middle");
          text.setAttribute("fill", "#667085");
          text.setAttribute("font-size", "10.5");
          text.setAttribute("font-weight", "650");
          text.setAttribute(
            "font-family",
            "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
          );
          text.textContent = statText;
          statsGroup.appendChild(text);
        });

        clone.appendChild(statsGroup);
      }

      const serializer = new XMLSerializer();
      const svgText = serializer.serializeToString(clone);
      const svgBlob = new Blob([svgText], {
        type: "image/svg+xml;charset=utf-8",
      });
      const url = URL.createObjectURL(svgBlob);

      try {
        const image = new Image();
        image.decoding = "async";
        image.src = url;
        await image.decode();

        const canvas = document.createElement("canvas");
        canvas.width = Math.ceil(width * scale);
        canvas.height = Math.ceil(exportHeight * scale);

        const context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);

        return await new Promise((resolve, reject) => {
          canvas.toBlob((blob) => {
            if (blob) {
              resolve(blob);
            } else {
              reject(new Error("Could not create PNG"));
            }
          }, "image/png");
        });
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    function downloadPngBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename || "chart.png";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    async function copyChartAsPng(button) {
      const block = button.closest(".chart-block");
      const svg = block.querySelector("svg");
      const stats = block.querySelector(".stats");
      const status = block.querySelector(".copy-status");

      button.disabled = true;
      status.textContent = "Copying...";

      try {
        const blob = await svgToPngBlob(svg, stats);
        if (!navigator.clipboard || !window.ClipboardItem) {
          downloadPngBlob(blob, button.dataset.filename);
          status.textContent = "Downloaded";
          return;
        }

        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]);
        status.textContent = "Copied";
      } catch (error) {
        console.error(error);
        status.textContent = "Copy failed";
      } finally {
        button.disabled = false;
        window.setTimeout(() => {
          if (
            status.textContent === "Copied" ||
            status.textContent === "Downloaded"
          ) {
            status.textContent = "";
          }
        }, 2200);
      }
    }

    document.querySelectorAll(".copy-chart").forEach((button) => {
      button.addEventListener("click", () => copyChartAsPng(button));
    });
"""


def filename_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "chart"


def copy_chart_button(label: str, filename: str | None = None) -> str:
    filename_attr = f' data-filename="{escape(filename)}"' if filename else ""

    return (
        '<div class="chart-actions">'
        f'<button class="copy-chart" type="button"{filename_attr} '
        f'aria-label="Copy {escape(label)} as PNG">'
        "Copy PNG</button>"
        '<span class="copy-status" aria-live="polite"></span>'
        "</div>"
    )
