from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ENGINE_LABELS = {
    "llmeng": "llm-engine",
    "minisgl": "mini-sglang",
}

METRIC_OPTIONS = (
    {
        "value": "ttft",
        "label": "the first token",
        "chart_label": "Time To First Token",
        "metric_name": "time_to_first_token_ms",
        "unit": "ms",
    },
    {
        "value": "ttlt",
        "label": "the last token",
        "chart_label": "Time To Last Token",
        "metric_name": "request_latency",
        "unit": "s",
    },
    {
        "value": "itl",
        "label": "each token",
        "chart_label": "Inter-Token Latency",
        "metric_name": "inter_token_latency_ms",
        "unit": "ms",
    },
)

AGGREGATE_OPTIONS = (
    {"value": "median", "label": "median"},
    {"value": "p90", "label": "p90"},
    {"value": "p95", "label": "p95"},
)

SLO_OPTIONS = (
    {"value": 10, "label": "10 ms"},
    {"value": 30, "label": "30 ms"},
    {"value": 100, "label": "100 ms"},
    {"value": 300, "label": "300 ms"},
    {"value": 1000, "label": "1 second"},
    {"value": 3000, "label": "3 seconds"},
    {"value": 10000, "label": "10 seconds"},
    {"value": 30000, "label": "30 seconds"},
    {"value": 60000, "label": "1 minute"},
)

WORKLOAD_OPTIONS = (
    {"value": "128;1024", "label": "128 tokens in / 1024 tokens out"},
    {"value": "256;2048", "label": "256 tokens in / 2048 tokens out"},
    {"value": "512;512", "label": "512 tokens in / 512 tokens out"},
    {"value": "512;4096", "label": "512 tokens in / 4096 tokens out"},
    {"value": "1024;128", "label": "1024 tokens in / 128 tokens out"},
    {"value": "1024;1024", "label": "1024 tokens in / 1024 tokens out"},
    {"value": "2048;256", "label": "2048 tokens in / 256 tokens out"},
    {"value": "2048;2048", "label": "2048 tokens in / 2048 tokens out"},
    {"value": "4096;512", "label": "4096 tokens in / 512 tokens out"},
)

SHOW_OPTIONS = (
    {"value": "throughput", "label": "the highest throughput"},
    {"value": "latency", "label": "the lowest latency"},
    {"value": "all", "label": "every benchmarked"},
)

METRIC_OPTION_BY_VALUE = {option["value"]: option for option in METRIC_OPTIONS}
AGGREGATE_VALUES = tuple(option["value"] for option in AGGREGATE_OPTIONS)

DEFAULT_METRIC = "ttft"
DEFAULT_AGGREGATE = "p95"
DEFAULT_SLO_MS = 1000
DEFAULT_VIEW = "all"
DEFAULT_WORKLOAD = "128;1024"


def _metric_seconds(
    result: dict[str, Any], metric_value: str, aggregate: str
) -> float | None:
    metric_meta = METRIC_OPTION_BY_VALUE.get(metric_value)
    if metric_meta is None:
        raise ValueError(f"Unsupported metric: {metric_value}")
    successful = (
        result.get("metrics", {})
        .get(metric_meta["metric_name"], {})
        .get("successful", {})
    )
    if aggregate == "median":
        value = successful.get("median")
    else:
        value = successful.get("percentiles", {}).get(aggregate)
    if value is None:
        return None
    value = float(value)
    if value is None:
        return None
    return value / 1000.0 if metric_meta["unit"] == "ms" else value


def extract_chart_points(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for result in results:
        request_totals = result.get("request_totals") or result.get("metrics", {}).get(
            "request_totals", {}
        )
        if int(request_totals.get("successful", 0)) <= 0:
            continue

        match = re.search(
            r"prompt_tokens=(\d+),output_tokens=(\d+)",
            str(result.get("data") or result.get("request_loader", {}).get("data", "")),
        )
        if match is None:
            continue
        prompt_tokens, output_tokens = int(match.group(1)), int(match.group(2))

        rps = result.get("metrics", {}).get("requests_per_second", {})
        successful = rps.get("successful", {})
        value = successful.get("mean")
        if value is None:
            continue
        n_gpu = int(result.get("n_gpu", 1))
        if n_gpu <= 0:
            continue
        throughput = float(value) / n_gpu

        metrics = {
            f"{metric_value}_{aggregate}": _metric_seconds(
                result, metric_value, aggregate
            )
            for metric_value in METRIC_OPTION_BY_VALUE
            for aggregate in AGGREGATE_VALUES
        }
        if any(value is None or value <= 0 for value in metrics.values()):
            continue

        worker = result.get("worker", {})
        model = str(result.get("model") or worker.get("backend_model") or "unknown")
        engine = str(result.get("engine", "unknown"))
        gpu_type = str(result.get("gpu_type", "unknown"))
        n_gpu = int(result.get("n_gpu", 1))
        points.append(
            {
                "engine": engine,
                "engine_label": ENGINE_LABELS.get(engine, engine),
                "model": model,
                "model_label": model.split("/")[-1].replace("-", " "),
                "gpu_type": gpu_type,
                "gpu_label": gpu_type.upper(),
                "n_gpu": n_gpu,
                "workload_value": f"{prompt_tokens};{output_tokens}",
                "workload_label": (
                    f"{prompt_tokens} tokens in / {output_tokens} tokens out"
                ),
                "rate_type": str(result.get("rate_type", "")),
                "throughput_per_replica": throughput,
                **metrics,
            }
        )
    return points


def _build_filter_options(
    points: list[dict[str, Any]],
    value_key: str,
    label_key: str,
    default_label: str,
) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for point in points:
        seen.setdefault(point[value_key], point[label_key])
    return [{"value": "all", "label": default_label}] + [
        {"value": value, "label": label} for value, label in sorted(seen.items())
    ]


def render_html(points: list[dict[str, Any]]) -> str:
    model_options = _build_filter_options(points, "model", "model_label", "any model")
    engine_options = _build_filter_options(
        points, "engine", "engine_label", "any engine"
    )
    payload = {
        "points": points,
        "modelOptions": model_options,
        "engineOptions": engine_options,
        "metricOptions": list(METRIC_OPTIONS),
        "aggregateOptions": list(AGGREGATE_OPTIONS),
        "workloadOptions": list(WORKLOAD_OPTIONS),
        "sloOptions": list(SLO_OPTIONS),
        "showOptions": list(SHOW_OPTIONS),
        "defaults": {
            "model": model_options[1]["value"] if len(model_options) > 1 else "all",
            "engine": "all",
            "metric": DEFAULT_METRIC,
            "aggregate": DEFAULT_AGGREGATE,
            "sloMs": DEFAULT_SLO_MS,
            "workload": DEFAULT_WORKLOAD,
            "show": DEFAULT_VIEW,
        },
    }

    data_json = json.dumps(payload, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Benchmark Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f8f7f3;
      --panel: #fcfbf8;
      --border: #cfc9bc;
      --text: #2d3a2a;
      --muted: #647062;
      --accent: #51623b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 40px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 22px 24px;
      text-align: center;
      line-height: 1.8;
      font-size: 20px;
    }}
    .card.small {{
      font-size: 17px;
      line-height: 1.6;
    }}
    select {{
      font: inherit;
      color: var(--text);
      background: #eef0e8;
      border: 1px solid #a7ad9f;
      border-radius: 6px;
      padding: 6px 12px;
      min-width: 0;
    }}
    .chart-wrap {{
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.96));
      border: 1px solid var(--border);
      padding: 20px 18px 8px;
    }}
    .status {{
      min-height: 22px;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
    }}
    .canvas-shell {{
      position: relative;
      width: 100%;
      height: 560px;
    }}
    canvas {{
      display: block;
      width: 100% !important;
      height: 100% !important;
    }}
    .footer-controls {{
      display: flex;
      justify-content: center;
      gap: 28px;
      align-items: center;
      margin-top: 14px;
      color: var(--accent);
      font-size: 15px;
    }}
    .footer-controls label {{
      display: inline-flex;
      gap: 10px;
      align-items: center;
    }}
    @media (max-width: 900px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      .footer-controls {{
        flex-direction: column;
        gap: 12px;
      }}
      .card {{
        font-size: 18px;
      }}
      .canvas-shell {{
        height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="controls">
      <div class="card">
        I want to serve
        <select id="model-select"></select>
        <br>
        with
        <select id="engine-select"></select>
      </div>
      <div class="card">
        Clients should receive
        <select id="latency-select"></select>
        <br>
        in under
        <select id="slo-select"></select>
        95% of the time
      </div>
      <div class="card">
        I expect on average
        <br>
        <select id="workload-select"></select>
      </div>
      <div class="card">
        I want to see
        <select id="show-select"></select>
        <br>
        configuration
      </div>
    </div>

    <div class="chart-wrap">
      <p id="status" class="status"></p>
      <div class="canvas-shell">
        <canvas id="benchmark-chart"></canvas>
      </div>
      <div class="footer-controls">
        <label>
          Metric:
          <select id="metric-select"></select>
        </label>
        <label>
          Aggregate:
          <select id="aggregate-select"></select>
        </label>
      </div>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const palette = [
      "#c76374",
      "#6271cc",
      "#657d37",
      "#c88c4a",
      "#7d5ab5",
      "#4d8c8f"
    ];

    const modelSelect = document.getElementById("model-select");
    const engineSelect = document.getElementById("engine-select");
    const latencySelect = document.getElementById("latency-select");
    const metricSelect = document.getElementById("metric-select");
    const sloSelect = document.getElementById("slo-select");
    const workloadSelect = document.getElementById("workload-select");
    const showSelect = document.getElementById("show-select");
    const aggregateSelect = document.getElementById("aggregate-select");
    const status = document.getElementById("status");

    function populateSelect(select, options, selectedValue) {{
      for (const option of options) {{
        const el = document.createElement("option");
        el.value = option.value;
        el.textContent = option.label;
        if (String(option.value) === String(selectedValue)) {{
          el.selected = true;
        }}
        select.appendChild(el);
      }}
    }}

    populateSelect(modelSelect, DATA.modelOptions, DATA.defaults.model);
    populateSelect(engineSelect, DATA.engineOptions, DATA.defaults.engine);
    populateSelect(latencySelect, DATA.metricOptions, DATA.defaults.metric);
    populateSelect(metricSelect, DATA.metricOptions.map((option) => ({{
      value: option.value,
      label: option.chart_label
    }})), DATA.defaults.metric);
    populateSelect(aggregateSelect, DATA.aggregateOptions, DATA.defaults.aggregate);
    populateSelect(sloSelect, DATA.sloOptions, DATA.defaults.sloMs);
    populateSelect(workloadSelect, DATA.workloadOptions, DATA.defaults.workload);
    populateSelect(showSelect, DATA.showOptions, DATA.defaults.show);

    const metricMeta = Object.fromEntries(DATA.metricOptions.map((option) => [option.value, option]));
    const chart = new Chart(document.getElementById("benchmark-chart"), {{
      type: "line",
      data: {{ datasets: [] }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {{ mode: "nearest", intersect: false }},
        plugins: {{
          legend: {{
            position: "bottom",
            labels: {{
              usePointStyle: true,
              boxWidth: 10,
              font: {{ family: '"Iowan Old Style", "Palatino Linotype", Georgia, serif', size: 14 }}
            }}
          }},
          tooltip: {{
            callbacks: {{
              label(context) {{
                const value = context.raw;
                return [
                  `framework: ${{context.dataset.label}}`,
                  `latency: ${{value.y.toFixed(3)}} s`,
                  `throughput: ${{value.x.toFixed(2)}} req/s`
                ];
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            type: "linear",
            title: {{
              display: true,
              text: ["Throughput (requests/s per replica)", "Higher is better ->"],
              color: "#51623b",
              font: {{ family: '"Iowan Old Style", "Palatino Linotype", Georgia, serif', size: 18 }}
            }},
            ticks: {{
              callback(value) {{
                return Number(value).toFixed(1);
              }}
            }},
            grid: {{
              color: "rgba(125, 129, 112, 0.18)"
            }}
          }},
          y: {{
            type: "logarithmic",
            min: 0.001,
            title: {{
              display: true,
              text: ["Latency (s)", "<- Lower is better"],
              color: "#51623b",
              font: {{ family: '"Iowan Old Style", "Palatino Linotype", Georgia, serif', size: 18 }}
            }},
            ticks: {{
              callback(value) {{
                const numeric = Number(value);
                if (numeric >= 1) return numeric.toFixed(0);
                return numeric.toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
              }}
            }},
            grid: {{
              color: "rgba(125, 129, 112, 0.18)"
            }}
          }}
        }}
      }}
    }});

    function syncMetricSelections(source) {{
      const value = source.value;
      latencySelect.value = value;
      metricSelect.value = value;
    }}

    function groupKey(point) {{
      return [point.engine, point.gpu_type, point.n_gpu, point.model].join("|");
    }}

    function maybeBestPoints(points, mode, metricKey) {{
      if (mode === "throughput") {{
        return [points.reduce((best, point) => point.throughput_per_replica > best.throughput_per_replica ? point : best)];
      }}
      if (mode === "latency") {{
        return [points.reduce((best, point) => point[metricKey] < best[metricKey] ? point : best)];
      }}
      return points;
    }}

    function seriesLabel(point, includeModel) {{
      const base = `${{point.engine_label}}, ${{point.n_gpu}} x ${{point.gpu_label}}`;
      return includeModel ? `${{base}}, ${{point.model_label}}` : base;
    }}

    function render() {{
      syncMetricSelections(latencySelect);
      const metricKey = metricSelect.value;
      const aggregateKey = aggregateSelect.value;
      const valueKey = `${{metricKey}}_${{aggregateKey}}`;
      const selectedModel = modelSelect.value;
      const selectedEngine = engineSelect.value;
      const selectedWorkload = workloadSelect.value;
      const selectedShow = showSelect.value;
      const visiblePoints = DATA.points.filter((point) => {{
        return (selectedModel === "all" || point.model === selectedModel)
          && (selectedEngine === "all" || point.engine === selectedEngine)
          && point.workload_value === selectedWorkload;
      }});

      if (visiblePoints.length === 0) {{
        chart.data.datasets = [];
        chart.update();
        status.textContent = "No benchmark points match the selected filters.";
        return;
      }}

      const includeModelInLegend = new Set(visiblePoints.map((point) => point.model)).size > 1;
      const grouped = new Map();
      for (const point of visiblePoints) {{
        const key = groupKey(point);
        const group = grouped.get(key) ?? [];
        group.push(point);
        grouped.set(key, group);
      }}

      const datasets = [];
      let colorIndex = 0;
      for (const points of grouped.values()) {{
        points.sort((left, right) => left.throughput_per_replica - right.throughput_per_replica);
        const chosen = maybeBestPoints(points, selectedShow, valueKey);
        datasets.push({{
          label: seriesLabel(points[0], includeModelInLegend),
          data: chosen.map((point) => ({{
            x: point.throughput_per_replica,
            y: point[valueKey]
          }})),
          borderColor: palette[colorIndex % palette.length],
          backgroundColor: palette[colorIndex % palette.length],
          pointRadius: 4,
          pointHoverRadius: 5,
          borderWidth: 3,
          showLine: chosen.length > 1,
          tension: 0
        }});
        colorIndex += 1;
      }}

      chart.data.datasets = datasets;
      chart.options.scales.y.title.text = [
        `${{metricMeta[metricKey].chart_label}} (s)`,
        "<- Lower is better"
      ];
      chart.update();
    }}

    latencySelect.addEventListener("change", () => {{
      syncMetricSelections(latencySelect);
      render();
    }});
    metricSelect.addEventListener("change", () => {{
      syncMetricSelections(metricSelect);
      render();
    }});
    modelSelect.addEventListener("change", render);
    engineSelect.addEventListener("change", render);
    sloSelect.addEventListener("change", render);
    workloadSelect.addEventListener("change", render);
    showSelect.addEventListener("change", render);
    aggregateSelect.addEventListener("change", render);

    render();
  </script>
</body>
</html>
"""


def convert_json_to_html(results: list[dict[str, Any]], output_path: Path) -> None:
    points = extract_chart_points(results)
    if not points:
        raise ValueError("No successful benchmark points found in inputs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(points))
    print(f"Wrote {len(points)} chart points to {output_path}")
