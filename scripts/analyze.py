"""Resume los resultados de results/ en las figuras y tablas de la memoria.

    python scripts/analyze.py q1        # una pregunta
    python scripts/analyze.py all       # todas

Junta las ejecuciones que solo se diferencian en la semilla y, para cada pregunta, guarda:

- memoria/figuras/<q>_accuracy.png: accuracy por ronda, media ± desviación típica entre semillas;
- memoria/figuras/<q>_accuracy_mb.png (Q3 y Q5): accuracy frente a MB transmitidos;
- memoria/tablas/<q>.md: una fila por configuración con
    · accuracy final: media de test_acc en las últimas 5 rondas,
    · oscilación: desviación típica de test_acc en las últimas 10 rondas,
    · salto medio: media de |acc(t) - acc(t-1)| en las últimas 10 rondas,
    · MB transmitidos en total (subida + bajada); "—" en el centralizado, que no se comunica.
  Cada métrica se calcula por ejecución y se da su media ± desviación típica (muestral) entre semillas.

Las ejecuciones que todavía no están completas se ignoran y se avisa de cuántas faltan.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments import EXPERIMENTS  # noqa: E402
from sweep import RESULTS_DIR, ROOT, collect_runs, is_done  # noqa: E402

FIGURES_DIR = ROOT / "memoria" / "figuras"
TABLES_DIR = ROOT / "memoria" / "tablas"

# Paleta categórica en orden fijo: el color sigue a la configuración, no a su posición.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Para cada pregunta: título, qué separa los paneles (None = un solo panel), qué distingue las
# curvas dentro de un panel y si se dibuja también la accuracy frente a los MB transmitidos.
LAYOUT = {
    "q1": ("Heterogeneidad de los datos (MLP)", None, "data", False),
    "q2": ("Épocas locales", "data", "epochs", False),
    "q3": ("Fracción de clientes por ronda (Dirichlet α=0,1)", None, "fraction", True),
    "q4": ("FedAvg frente a FedProx", "scenario", "method", False),
    "q5": ("Cuantización int8 de la subida", "data", "codec", True),
    "q6": ("CNN frente a MLP", "data", "model", False),
    "cnn": ("Variantes de la CNN", "data", "model", False),
    "topk": ("Compresión de la subida: int8 frente a top-k", "data", "codec", True),
    "clientes": ("Número de clientes (Dirichlet α=0,1, 5 por ronda)", None, "clients", False),
}

MODEL_NAMES = {"mlp": "MLP", "cnn": "CNN", "cnn3x3": "CNN 3×3", "cnn_gn": "CNN + GroupNorm"}
CODEC_NAMES = {"none": "float32", "int8": "int8", "topk": "top-k"}

LAST_FINAL, LAST_STABILITY = 5, 10


def num(x, digits=None):
    """Número con coma decimal, como en el resto de la memoria."""
    text = f"{x:.{digits}f}" if digits is not None else f"{x:g}"
    return text.replace(".", ",")


def describe(cfg):
    """Etiquetas legibles de una configuración (config.json de la ejecución)."""
    p, c = cfg["partition"], cfg["client"]
    if cfg["mode"] == "local_only":
        data = f"Solo local (α={num(p['alpha'])})"
    elif p["n_clients"] == 1:
        data = "Centralizado"
    elif p["type"] == "iid":
        data = "IID"
    elif p["type"] == "dirichlet":
        data = f"Dirichlet α={num(p['alpha'])}"
    else:
        data = f"k={p['classes_per_client']} clases"
    mu = c["fedprox_mu"]
    return {
        "data": data,
        "epochs": f"E={c['local_epochs']}",
        "fraction": f"C={num(cfg['server']['fraction'])}",
        "method": "FedAvg" if mu == 0 else f"FedProx μ={num(mu)}",
        "scenario": data + (f", E={c['local_epochs']}" if c["local_epochs"] != 1 else ""),
                "codec": CODEC_NAMES.get(cfg["server"]["codec"], cfg["server"]["codec"])
        + (f" ({num(cfg['server']['topk_fraction'] * 100)} %)" if cfg["server"]["codec"] == "topk" else ""),
        "clients": f"K={p['n_clients']}, C={num(cfg['server']['fraction'])}",
        "model": MODEL_NAMES.get(cfg["model"]["name"], cfg["model"]["name"]),
    }


def read_metrics(folder):
    with open(ROOT / RESULTS_DIR / folder / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {col: np.array([float(r[col]) for r in rows]) for col in ("test_acc", "bytes_cum")}


def load_question(name):
    """Agrupa por configuración (el nombre de la carpeta sin la semilla) las ejecuciones hechas."""
    groups, missing = {}, 0
    for folder, _, rounds in collect_runs([name]):
        if not is_done(folder, rounds):
            missing += 1
            continue
        key = re.sub(r"_s\d+$", "", folder)
        if key not in groups:
            cfg = json.loads((ROOT / RESULTS_DIR / folder / "config.json").read_text())
            groups[key] = {"labels": describe(cfg), "runs": []}
        groups[key]["runs"].append(read_metrics(folder))
    return list(groups.values()), missing


def summary(runs):
    """Media y desviación típica entre semillas de cada métrica de la tabla."""
    per_run = {"final": [], "oscillation": [], "jump": [], "mb": []}
    for r in runs:
        acc, tail = r["test_acc"], r["test_acc"][-LAST_STABILITY:]
        per_run["final"].append(acc[-LAST_FINAL:].mean())
        per_run["oscillation"].append(tail.std())
        per_run["jump"].append(np.abs(np.diff(tail)).mean())
        per_run["mb"].append(r["bytes_cum"][-1] / 1e6)
    return {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) for k, v in per_run.items()}


def mean_std(text_mean, text_std):
    return text_mean if text_std is None else f"{text_mean} ± {text_std}"


def write_table(name, groups, panel_key, curve_key):
    header = (["Escenario"] if panel_key else []) + [
        "Configuración", "Semillas", "Accuracy final", "Oscilación", "Salto medio", "MB",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for g in groups:
        s, n = summary(g["runs"]), len(g["runs"])

        def cell(key, digits):
            m, sd = s[key]
            return mean_std(num(m, digits), None if np.isnan(sd) else num(sd, digits))

        row = ([g["labels"][panel_key]] if panel_key else []) + [
            g["labels"][curve_key], str(n), cell("final", 4), cell("oscillation", 4), cell("jump", 4),
            "—" if g["labels"]["data"] == "Centralizado" else num(s["mb"][0], 1),
        ]
        lines.append("| " + " | ".join(row) + " |")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"{name}.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def plot(name, groups, panel_key, curve_key, x="round"):
    title = LAYOUT[name][0]
    panels = list(dict.fromkeys(g["labels"][panel_key] for g in groups)) if panel_key else [None]
    curves = list(dict.fromkeys(g["labels"][curve_key] for g in groups))
    color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(curves)}

    ncols = 2 if len(panels) == 4 else min(3, len(panels))
    nrows = -(-len(panels) // ncols)
    width = 7.5 if ncols == 1 else 5 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.6 * nrows + 1.2), sharey=True, squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        for g in groups:
            if panel_key and g["labels"][panel_key] != panel:
                continue
            acc = np.array([r["test_acc"] for r in g["runs"]])
            mean, std = acc.mean(axis=0), acc.std(axis=0, ddof=1) if len(acc) > 1 else 0 * acc[0]
            if x == "round":
                xs = np.arange(1, len(mean) + 1)
            else:
                xs = np.mean([r["bytes_cum"] for r in g["runs"]], axis=0) / 1e6
            label = g["labels"][curve_key]
            ax.plot(xs, mean, color=color[label], linewidth=2, label=label)
            ax.fill_between(xs, mean - std, mean + std, color=color[label], alpha=0.15, linewidth=0)
        if panel:
            ax.set_title(panel, fontsize=11)
        ax.set_xlabel("ronda" if x == "round" else "MB transmitidos (acumulado)")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (test)")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in list(axes.flat)[1:len(panels)]:  # una entrada por curva aunque no esté en el primer panel
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    # Cabecera con medidas fijas en pulgadas: título, subtítulo y leyenda encima de los paneles.
    legend_cols = min(len(labels), 3 if ncols == 1 else 6)
    legend_rows = -(-len(labels) // legend_cols)
    height = fig.get_figheight()
    header = 0.75 + 0.28 * legend_rows
    fig.suptitle(title, fontsize=12, y=1 - 0.12 / height, va="top")
    fig.text(0.5, 1 - 0.42 / height, "media ± desviación típica entre semillas",
             ha="center", va="top", fontsize=9, color="#555555")
    fig.legend(handles, labels, loc="upper center", ncol=legend_cols, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.62 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (f"{name}_accuracy.png" if x == "round" else f"{name}_accuracy_mb.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    args = parser.parse_args()
    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments

    for name in names:
        groups, missing = load_question(name)
        print(f"\n== {name}: {LAYOUT[name][0]}")
        if missing:
            print(f"   (faltan {missing} ejecuciones; se resume lo que hay)")
        if not groups:
            print("   sin resultados todavía")
            continue
        _, panel_key, curve_key, vs_mb = LAYOUT[name]
        print(write_table(name, groups, panel_key, curve_key))
        print("   ->", plot(name, groups, panel_key, curve_key).relative_to(ROOT))
        if vs_mb:
            print("   ->", plot(name, groups, panel_key, curve_key, x="mb").relative_to(ROOT))


if __name__ == "__main__":
    main()
