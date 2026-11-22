"""Figures for the comparison and the redirect, from the stored results.

Every plotted number is read from `out/` (or `results/`), never typed here:

  out/tables_main.json          removal: report refusal, KL, perplexity
  out/holdout_<model>.json      removal: held-out refusal
  out/bench_<name>.json         removal: capability tasks
  out/timing.json               removal: stage timings and Heretic wall clock
  out/redirect/gradient_summary.json   the -r and acceptance-gradient sweeps
  out/redirect/pilot_probe.json        removal and dense controls
  out/redirect/kl_split.json           intended vs unintended KL
  out/redirect/category_cyber_*.json   the category control
  out/graph/graph_*.json               same nodes, new edges

Writes PNGs and `numbers.md` to `docs/figures/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from libershift.constants import OUT, ROOT

# Written into docs/ so the README's images are regenerated in place.
FIGURES = ROOT / "docs" / "figures"
MODELS = {
    "1.5B": {
        "title": "Qwen2.5-1.5B-Instruct",
        "table": "tables_main.json",
        "holdout": "Qwen2.5-1.5B-Instruct",
        "bench": {"base": "bench_base.json", "taa": "bench_taa.json",
                  "shipped": "bench_taa_target.json",
                  "heretic_mpoa": "bench_heretic_mpoa.json",
                  "heretic_ara": "bench_heretic_ara.json"},
    },
    "0.5B": {
        "title": "Qwen2.5-0.5B-Instruct",
        "table": "tables__05b.json",
        "holdout": "Qwen2.5-0.5B-Instruct",
        "bench": {"base": "bench_base_05b.json", "taa": "bench_taa_05b.json",
                  "shipped": "bench_taa_target_05b.json",
                  "heretic_mpoa": "bench_heretic_mpoa_05b.json",
                  "heretic_ara": "bench_heretic_ara_05b.json"},
    },
}
LABELS = {"base": "base", "taa": "LiberShift", "dense_removal": "dense removal",
          "shipped": "LiberShift+target", "heretic_mpoa": "Heretic MPOA",
          "heretic_ara": "Heretic ARA"}
COLOURS = {"base": "#666666", "taa": "#1b6ca8", "shipped": "#0f4c75",
           "heretic_mpoa": "#c46a1a", "heretic_ara": "#a8201a"}
MARKERS = {"base": "s", "taa": "o", "shipped": "D", "heretic_mpoa": "^",
           "heretic_ara": "v"}

# The shipped configuration in each family file: dense removal with the target on
# the efficient few.  Loaded from the family so the figure cannot drift from it.
SHIPPED = {"1.5B": ("redirect/dense_hybrid_family.json", "dense_q0.95_g0.45"),
           "0.5B": ("redirect/dense_hybrid_05b_family.json", "dense_q0.95_g0.6")}


RESULTS = OUT


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def name_for(model_key: str) -> str:
    return ("dense_hybrid_responses.json" if model_key == "1.5B"
            else "dense_hybrid_05b_responses.json")


def gather() -> dict:
    out = {}
    for key, cfg in MODELS.items():
        table = load(cfg["table"])
        holdout = load(f"holdout_{cfg['holdout']}.json")
        bench = {}
        for model, filename in cfg["bench"].items():
            if (RESULTS / filename).exists():
                bench[model] = load(filename)["tasks"]
        path, label = SHIPPED[key]
        point = load(path)["results"][label]
        responses = load(f"redirect/{name_for(key)}")["results"][label]["responses"]
        report = responses[30:]  # the 30 calibration prompts are excluded
        shipped = {"kl_divergence": point["kl_divergence"],
                   "refusal_rate": point["heldout_300"]["refusal_rate"],
                   "refusals": point["heldout_300"]["refusals"],
                   "n": point["heldout_300"]["n"],
                   "ppl": point["perplexity_wikitext2"],
                   "report_rate": sum(r["refusal"] for r in report) / len(report),
                   "report_refusals": sum(r["refusal"] for r in report)}
        out[key] = {"cfg": cfg, "table": table, "holdout": holdout, "bench": bench,
                    "shipped": shipped}
    return out


def tradeoff(data: dict) -> None:
    """Refusal against KL, one panel per model, with direct labels.

    No legend and no in-plot table: the five points are named where they sit, so
    the reader does not have to match a marker to a row.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharey=True)
    order = ("base", "taa", "shipped", "heretic_mpoa", "heretic_ara")
    # Label offsets in points, tuned per panel so nothing overlaps.
    offsets = {
        "1.5B": {"base": (12, -3), "taa": (-86, -3), "shipped": (12, -14),
                 "heretic_mpoa": (-30, 13), "heretic_ara": (-16, 13)},
        "0.5B": {"base": (12, -3), "taa": (-10, 14), "shipped": (12, -14),
                 "heretic_mpoa": (-86, 4), "heretic_ara": (-16, 13)},
    }
    names = {"base": "base", "taa": "dense removal", "shipped": "+ target 5%",
             "heretic_mpoa": "Heretic MPOA", "heretic_ara": "Heretic ARA"}
    for ax, (key, d) in zip(axes, data.items()):
        for model in order:
            if model == "shipped":
                kl = d["shipped"]["kl_divergence"]
                refusal = d["shipped"]["refusal_rate"]
            else:
                kl = d["table"][model]["kl_divergence"]
                refusal = d["holdout"][model]["refusal_rate"]
            x = max(kl, 5e-4)
            ax.scatter(x, refusal, s=105, color=COLOURS[model],
                       marker=MARKERS[model], zorder=3,
                       facecolors="none" if model == "shipped" else COLOURS[model],
                       linewidths=1.8)
            ax.annotate(names[model], (x, refusal), textcoords="offset points",
                        xytext=offsets[key][model], fontsize=9.5,
                        color=COLOURS[model], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=COLOURS[model],
                                        shrinkA=2, shrinkB=4))
        ax.set_xscale("log")
        ax.set_xlim(4e-4, 9)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel("KL divergence from base (lower is better)")
        ax.set_title(d["cfg"]["title"], fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelsize=9.5)
    axes[0].set_ylabel("refusal rate, 300 held-out harmful prompts", fontsize=10)
    fig.suptitle("Refusal removed against damage: lower left is better",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1-refusal-vs-damage.png", dpi=160)
    plt.close(fig)


def refusal_bars(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    models = ["base", "taa", "shipped", "heretic_mpoa", "heretic_ara"]
    for ax, (key, d) in zip(axes, data.items()):
        values, counts = [], []
        for m in models:
            source = d["shipped"] if m == "shipped" else d["holdout"][m]
            values.append(source["refusal_rate"])
            counts.append(f"{source['refusals']}/{source['n']}")
        bars = ax.bar([LABELS[m].replace("+", "\n+") for m in models], values,
                      color=[COLOURS[m] for m in models])
        for bar, value, count in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                    f"{value:.3f}\n({count})", ha="center", fontsize=8)
        ax.set_title(d["cfg"]["title"], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelrotation=12)
    axes[0].set_ylabel("refusal rate, held-out prompts")
    fig.suptitle("Held-out refusal: 100 JailbreakBench + 200 BeaverTails prompts", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3-heldout-refusal.png", dpi=160)
    plt.close(fig)


def capability(data: dict) -> None:
    tasks = ["arc_easy", "piqa", "winogrande"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(12.4, 3.9), sharey=True)
    models = ["base", "taa", "shipped", "heretic_mpoa", "heretic_ara"]
    for ax, task in zip(axes, tasks):
        for i, key in enumerate(("1.5B", "0.5B")):
            d = data[key]
            xs, ys = [], []
            for j, model in enumerate(models):
                result = d["bench"].get(model, {}).get(task)
                if not result:
                    continue
                xs.append(j + (i - 0.5) * 0.18)
                ys.append(result["acc,none"])
            ax.bar(xs, ys, width=0.17, color="#1b6ca8" if i == 0 else "#7fb2d6",
                   label="1.5B" if i == 0 else "0.5B")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS[m].replace("+", "\n+") for m in models],
                           fontsize=8, rotation=12)
        ax.set_title(task, fontsize=10)
        ax.set_ylim(0.3, 0.8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("accuracy, 300 samples")
    axes[0].legend(fontsize=8)
    fig.suptitle("Capability retained after the edit", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9-capability.png", dpi=160)
    plt.close(fig)


def timing() -> None:
    t = load("timing.json")
    stages = t["stages"]
    names = ["load model", "capture 800 prompts", "attribute 28 layers",
             "calibrate 6 planes x 30 prompts", "apply edit",
             "export checkpoint + patch", "final evaluation (370 prompts)"]
    values = [stages[n] for n in names]
    heretic = t["heretic_clean_wall_seconds"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(range(len(names)), values, color="#1b6ca8")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    for i, value in enumerate(values):
        axes[0].text(value + 0.2, i, f"{value:.2f} s", va="center", fontsize=8)
    axes[0].set_xlabel("seconds")
    axes[0].set_title("LiberShift, stage by stage", fontsize=10)
    axes[0].grid(alpha=0.3, axis="x")

    labels = ["LiberShift\n(produce)", "LiberShift\n(+ one eval)",
              "Heretic MPOA\n40 trials", "Heretic ARA\n40 trials"]
    totals = [sum(values[:-1]), sum(values),
              heretic["mpoa_40_trials"], heretic["ara_40_trials"]]
    bars = axes[1].bar(labels, totals,
                       color=["#1b6ca8", "#7fb2d6", "#c46a1a", "#a8201a"])
    for bar, total in zip(bars, totals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, total + 5,
                     f"{total:.0f} s", ha="center", fontsize=9)
    axes[1].set_ylabel("wall clock, seconds")
    axes[1].set_title("total time to a usable model", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11-timing.png", dpi=160)
    plt.close(fig)


def redirect_pareto() -> None:
    """The novel edit: refusal against KL for both targets and the controls."""
    sweep = load("redirect/gradient_summary.json")["results"]
    probe = load("redirect/pilot_probe.json")
    series = [
        ("mirrored target (-r)", "neg_r", "#c46a1a", "^", "-"),
        ("acceptance gradient", "grad", "#1b6ca8", "o", "-"),
        ("acceptance gradient, orthogonalised", "grad_orth", "#7fb2d6", "s", "--"),
    ]
    controls = {
        "removal, 90% load": (probe["removal_s1"], "#222222", "s"),
        "dense removal": (probe["dense_removal"], "#a8201a", "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    panels = [
        ("refusal_100", "100-prompt evaluation slice"),
        ("heldout_300", "300 held-out prompts"),
    ]
    for ax, (field, title) in zip(axes, panels):
        for label, prefix, colour, marker, style in series:
            xs, ys, gammas = [], [], []
            for gamma, shown in (("0.3", "0.3"), ("0.5", "0.5"),
                                 ("0.75", "0.75"), ("1", "1.0")):
                key = f"{prefix}_g{gamma}"
                if key not in sweep:
                    continue
                xs.append(sweep[key]["kl_divergence"])
                ys.append(sweep[key][field]["refusal_rate"])
                gammas.append(shown)
            ax.plot(xs, ys, style, color=colour, marker=marker, lw=1.4,
                    ms=5, label=label)
            for x, y, gamma in zip(xs, ys, gammas):
                if prefix != "grad_orth" and gamma in ("0.3", "0.75"):
                    ax.annotate(f"$\\gamma$={gamma}", (x, y),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, color=colour)
        for label, (result, colour, marker) in controls.items():
            ax.scatter(result["kl_divergence"], result[field]["refusal_rate"],
                       color=colour, marker=marker, s=75, zorder=4, label=label)
        ax.set_xscale("log")
        ax.set_xticks([0.02, 0.03, 0.05, 0.1, 0.2])
        ax.set_xticklabels(["0.02", "0.03", "0.05", "0.10", "0.20"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.015, 0.23)
        ax.set_ylim(-0.03, 0.58)
        ax.set_xlabel("KL divergence from base")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("refusal rate")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("The redirect: refusal against KL (lower left is better)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4-redirect-pareto.png", dpi=160)
    plt.close(fig)


def redirect_kl_split() -> None:
    """How much of the redirect's KL is the behaviour it is asked to change."""
    configs = load("redirect/kl_split.json")["configs"]
    order = [
        ("removal_90", "removal, 90%"),
        ("neg_r_g0.5", "-r, $\\gamma$=0.5"),
        ("neg_r_g1", "-r, $\\gamma$=1.0"),
        ("grad_g0.3", "gradient, $\\gamma$=0.3"),
        ("grad_g0.5", "gradient, $\\gamma$=0.5"),
        ("grad_g0.75", "gradient, $\\gamma$=0.75"),
        ("dense_removal", "dense removal"),
    ]
    intended = [configs[key]["kl_in_accept_refuse_set"] for key, _ in order]
    elsewhere = [configs[key]["kl_outside_set"] for key, _ in order]
    totals = [configs[key]["kl_total"] for key, _ in order]
    shares = [configs[key]["in_set_share"] * 100 for key, _ in order]

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    rows = list(range(len(order)))
    ax.barh(rows, intended, color="#1b6ca8",
            label="intended: accept/refuse tokens")
    ax.barh(rows, elsewhere, left=intended, color="#cccccc",
            label="elsewhere")
    ax.set_yticks(rows)
    ax.set_yticklabels([label for _, label in order], fontsize=8)
    ax.invert_yaxis()
    for row, total, share in zip(rows, totals, shares):
        ax.text(total + 0.002, row, f"{total:.4f}  ({share:.1f}% intended)",
                va="center", fontsize=7.5)
    ax.set_xlim(0, max(totals) * 1.32)
    ax.set_xlabel("KL divergence (first token, 100 harmless prompts)")
    ax.set_title("Where the redirect's KL goes", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5-kl-split.png", dpi=160)
    plt.close(fig)


def redirect_category() -> None:
    """One category is selected; the others are the control."""
    files = [
        ("5k", "category_cyber_5k.json"),
        ("20k", "category_cyber_20k.json"),
        ("50k", "category_cyber_50k.json"),
    ]
    data = [load(f"redirect/{name}") for _, name in files]
    x = list(range(len(data)))
    width = 0.36
    cyber = [d["eval_cat"]["refusal_rate"] for d in data]
    other = [d["eval_other"]["refusal_rate"] for d in data]

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar([i - width / 2 for i in x], cyber, width, color="#1b6ca8",
           label="cyber (selected category)")
    ax.bar([i + width / 2 for i in x], other, width, color="#c46a1a",
           label="other categories")
    for i, d in enumerate(data):
        ax.text(i - width / 2, cyber[i] + 0.02,
                f"{cyber[i]:.3f}\n({d['eval_cat']['refusals']}/{d['eval_cat']['n']})",
                ha="center", fontsize=7.5)
        ax.text(i + width / 2, other[i] + 0.02,
                f"{other[i]:.3f}\n({d['eval_other']['refusals']}/{d['eval_other']['n']})",
                ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{label} atoms\nKL {d['all']['kl_divergence']:.3f}\n"
        f"strength {d['strength']:.1f}"
        for (label, _), d in zip(files, data)
    ], fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("refusal rate")
    ax.set_title("Category control: selected on cyber, measured on both",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7-category.png", dpi=160)
    plt.close(fig)


def graph_view() -> None:
    """The top-5 edge list moves while the gate walk stays fixed."""
    files = [
        ("$\\gamma$=0.3\nacceptance", "graph_grad_g0.3.json"),
        ("$\\gamma$=1.0\nacceptance", "graph_grad_g1.json"),
        ("$\\gamma$=1.0\nmirrored", "graph_neg_r_g1.json"),
    ]
    data = [load(f"graph/{name}") for _, name in files]
    x = list(range(len(data)))
    width = 0.36
    top = [d["target_changed"] for d in data]
    list_changed = [d["list_changed"] for d in data]
    walk_identical = all(d["walk"]["identical_before_after"] for d in data)
    walk_hits = {d["walk"]["hits"] for d in data}

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar([i - width / 2 for i in x], top, width, color="#7fb2d6",
           label="top target changed")
    ax.bar([i + width / 2 for i in x], list_changed, width, color="#1b6ca8",
           label="top-5 target list changed")
    for i in x:
        ax.text(i - width / 2, top[i] + 3, str(top[i]),
                ha="center", fontsize=8)
        ax.text(i + width / 2, list_changed[i] + 3, str(list_changed[i]),
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in files])
    ax.set_ylim(0, 215)
    ax.set_ylabel("features out of 200")
    ax.set_title("Same nodes, new edges", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    if walk_identical and len(walk_hits) == 1:
        hit_count = next(iter(walk_hits))
        ax.text(0.98, 0.97, f"walk: {hit_count}/{hit_count} hits identical\n"
                            "in all three settings",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#bbbbbb"))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8-graph-view.png", dpi=160)
    plt.close(fig)


def hybrid_sweep() -> None:
    """The dense family: removal for most atoms, redirect for the efficient few.

    Left, the result: held-out refusal against KL.  Right, the cost: KL against
    the number of atoms that carry the target component.  The second panel is the
    point of the rule, because the redirect's KL is driven by how many atoms get
    it, not by how many are removed.
    """
    family = load("redirect/dense_hybrid_family.json")["results"]
    sparse = load("redirect/hybrid_efficiency_fine.json")["results"]

    def series(gamma: float) -> list[dict]:
        return sorted((v for v in family.values()
                       if abs(v["gamma"] - gamma) < 1e-9 and v["quantile"] < 1.0),
                      key=lambda v: v["quantile"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for gamma, colour, marker in ((0.3, "#1b6ca8", "o"),
                                  (0.45, "#c46a1a", "s"),
                                  (0.6, "#7a4fa3", "^")):
        rows = series(gamma)
        if not rows:
            continue
        xs = [v["kl_divergence"] for v in rows]
        ys = [v["heldout_300"]["refusal_rate"] for v in rows]
        axes[0].plot(xs, ys, "-", color=colour, marker=marker, lw=1.0,
                     alpha=0.65, ms=5, label=f"$\\gamma$={gamma:g}")
        axes[1].plot([v["redirect_atoms"] for v in rows], xs, "-", color=colour,
                     marker=marker, lw=1.0, alpha=0.65, ms=5,
                     label=f"$\\gamma$={gamma:g}")

    control = family["dense_q1_g0"]
    axes[0].scatter(control["kl_divergence"],
                    control["heldout_300"]["refusal_rate"],
                    color="#222222", marker="D", s=85, zorder=5,
                    label="dense removal")
    axes[1].axhline(control["kl_divergence"], color="#222222", ls=":", lw=1.0)
    axes[1].annotate("dense removal", (4.5e4, control["kl_divergence"]),
                     textcoords="offset points", xytext=(0, 7), fontsize=7.5,
                     color="#222222")

    for index, key in enumerate(("efficiency_q0.25_g0.3", "efficiency_q0.3_g0.75")):
        row = sparse[key]
        axes[0].scatter(row["kl_divergence"], row["heldout_300"]["refusal_rate"],
                        color="#8a8a8a", marker="x", s=55, zorder=4,
                        label="35k sparse hybrid" if index == 0 else "_nolegend_")
        axes[1].scatter(row["redirect_atoms"], row["kl_divergence"],
                        color="#8a8a8a", marker="x", s=55, zorder=4,
                        label="35k sparse hybrid" if key.endswith("g0.3") else "_nolegend_")
    axes[0].annotate("35k sparse hybrid", (sparse["efficiency_q0.3_g0.75"]["kl_divergence"],
                     sparse["efficiency_q0.3_g0.75"]["heldout_300"]["refusal_rate"]),
                     textcoords="offset points", xytext=(-64, -14), fontsize=7.5,
                     color="#5a5a5a")

    chosen = family["dense_q0.95_g0.45"]
    axes[0].scatter(chosen["kl_divergence"], chosen["heldout_300"]["refusal_rate"],
                    facecolors="none", edgecolors="#c46a1a", s=220, lw=1.6, zorder=6)
    axes[0].annotate("q=0.95,\n$\\gamma$=0.45", (chosen["kl_divergence"],
                     chosen["heldout_300"]["refusal_rate"]),
                     textcoords="offset points", xytext=(-46, 10), fontsize=8,
                     color="#c46a1a")
    axes[1].scatter(chosen["redirect_atoms"], chosen["kl_divergence"],
                    facecolors="none", edgecolors="#c46a1a", s=220, lw=1.6, zorder=6)

    axes[0].set_xlim(0.022, 0.12)
    axes[0].set_ylim(-0.01, 0.12)
    axes[0].set_xlabel("KL divergence from base")
    axes[0].set_ylabel("refusal rate, 300 held-out prompts")
    axes[0].set_title("Held-out refusal against KL", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1e3, 3e3, 1e4, 3e4, 1e5])
    axes[1].set_xticklabels(["1k", "3k", "10k", "30k", "100k"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("atoms that carry the target component")
    axes[1].set_ylabel("KL divergence from base")
    axes[1].set_title("The cost is set by the redirect set, not the edit", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Dense removal plus a redirect of the efficient few", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2-where-the-target-goes.png", dpi=160)
    plt.close(fig)


def hybrid_05b() -> None:
    """The same family on the second model, same two panels as fig9."""
    family = load("redirect/dense_hybrid_05b_family.json")["results"]

    def series(gamma: float) -> list[dict]:
        return sorted((v for v in family.values()
                       if abs(v["gamma"] - gamma) < 1e-9 and v["quantile"] < 1.0),
                      key=lambda v: v["quantile"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for gamma, colour, marker in ((0.3, "#1b6ca8", "o"),
                                  (0.45, "#c46a1a", "s"),
                                  (0.6, "#7a4fa3", "^")):
        rows = series(gamma)
        if not rows:
            continue
        axes[0].plot([v["kl_divergence"] for v in rows],
                     [v["heldout_300"]["refusal_rate"] for v in rows],
                     "-", color=colour, marker=marker, lw=1.1, alpha=0.75, ms=5,
                     label=f"$\\gamma$={gamma:g}")
        axes[1].plot([v["redirect_atoms"] for v in rows],
                     [v["kl_divergence"] for v in rows],
                     "-", color=colour, marker=marker, lw=1.1, alpha=0.75, ms=5,
                     label=f"$\\gamma$={gamma:g}")

    control = family["dense_q1_g0"]
    axes[0].scatter(control["kl_divergence"], control["heldout_300"]["refusal_rate"],
                    color="#222222", marker="D", s=85, zorder=5, label="dense removal")
    axes[1].axhline(control["kl_divergence"], color="#222222", ls=":", lw=1.0)
    axes[1].annotate("dense removal", (2.5e4, control["kl_divergence"]),
                     textcoords="offset points", xytext=(0, 7), fontsize=7.5,
                     color="#222222")

    # The chosen point, marked as in figures 2 and 6.
    chosen = family["dense_q0.95_g0.6"]
    axes[0].scatter(chosen["kl_divergence"], chosen["heldout_300"]["refusal_rate"],
                    facecolors="none", edgecolors="#7a4fa3", s=220, lw=1.6, zorder=6,
                    label="chosen: q=0.95, $\\gamma$=0.6")
    axes[1].scatter(chosen["redirect_atoms"], chosen["kl_divergence"],
                    facecolors="none", edgecolors="#7a4fa3", s=220, lw=1.6, zorder=6)

    axes[0].set_xlim(0.075, 0.26)
    axes[0].set_ylim(0.08, 0.175)
    axes[0].set_xlabel("KL divergence from base")
    axes[0].set_ylabel("refusal rate, 300 held-out prompts")
    axes[0].set_title("Held-out refusal against KL", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1e3, 3e3, 1e4])
    axes[1].set_xticklabels(["1k", "3k", "10k"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("atoms that carry the target component")
    axes[1].set_ylabel("KL divergence from base")
    axes[1].set_title("The cost is set by the redirect set, not the edit", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("The same family on Qwen2.5-0.5B-Instruct", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10-second-model.png", dpi=160)
    plt.close(fig)


def frontier() -> None:
    """Held-out refusal against KL: this family, dense removal, and Heretic."""
    family = load("redirect/dense_hybrid_family.json")["results"]
    sparse = load("redirect/hybrid_efficiency_fine.json")["results"]
    tables = load("tables_main.json")
    holdout = load("holdout_Qwen2.5-1.5B-Instruct.json")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for gamma, colour in ((0.3, "#1b6ca8"), (0.45, "#c46a1a"), (0.6, "#7a4fa3")):
        rows = sorted((v for v in family.values()
                       if abs(v["gamma"] - gamma) < 1e-9),
                      key=lambda v: v["quantile"])
        ax.plot([v["kl_divergence"] for v in rows],
                [v["heldout_300"]["refusal_rate"] for v in rows],
                "-", color=colour, lw=1.0, alpha=0.55, marker="o", ms=5,
                label=f"dense removal + redirect, $\\gamma$={gamma:g}")

    points = [
        ("dense removal", family["dense_q1_g0"]["kl_divergence"],
         family["dense_q1_g0"]["heldout_300"]["refusal_rate"],
         "#222222", "D", (8, 6)),
        ("q=0.995, $\\gamma$=0.3", family["dense_q0.995_g0.3"]["kl_divergence"],
         family["dense_q0.995_g0.3"]["heldout_300"]["refusal_rate"],
         "#1b6ca8", "o", (8, 4)),
        ("35k hybrid, $\\gamma$=0.75", sparse["efficiency_q0.3_g0.75"]["kl_divergence"],
         sparse["efficiency_q0.3_g0.75"]["heldout_300"]["refusal_rate"],
         "#8a8a8a", "x", (6, -12)),
        ("Heretic MPOA", tables["heretic_mpoa"]["kl_divergence"],
         holdout["heretic_mpoa"]["refusal_rate"], "#a8201a", "v", (6, -13)),
        ("Heretic ARA", tables["heretic_ara"]["kl_divergence"],
         holdout["heretic_ara"]["refusal_rate"], "#a8201a", "P", (-74, 6)),
    ]
    for label, x, y, colour, marker, offset in points:
        ax.scatter(x, y, color=colour, marker=marker, s=85, zorder=4)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=8, color=colour)
    chosen = family["dense_q0.95_g0.45"]
    ax.scatter(chosen["kl_divergence"], chosen["heldout_300"]["refusal_rate"],
               facecolors="none", edgecolors="#c46a1a", s=230, lw=1.6, zorder=6,
               label="chosen: q=0.95, $\\gamma$=0.45")
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.03", "0.05", "0.10", "0.20", "0.50", "1.0"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.025, 1.2)
    ax.set_ylim(-0.01, 0.24)
    ax.set_xlabel("KL divergence from base")
    ax.set_ylabel("refusal rate, 300 held-out prompts")
    ax.set_title("Held-out refusal against KL", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6-frontier.png", dpi=160)
    plt.close(fig)

def qwen35_frontier() -> None:
    """Qwen3.5-4B: removal fails, the redirect still moves refusal.

    Every measured family point plus the measured Heretic checkpoints,
    held-out refusal against KL.  No lines: the grid is sparse and the reader
    should not infer interpolation.  The PR #211 reports live in the README
    prose, not on the chart.
    """
    family = load("redirect/dense_hybrid_35b.json")["results"]
    base = load("qwen35_base.json")["base_heldout_300"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    # Measured Heretic checkpoints, when the comparison file exists.
    comparison = None
    if (RESULTS / "comparison_Qwen3.5-4B.json").exists():
        comparison = load("comparison_Qwen3.5-4B.json")
        for name, marker, colour, label, dx, dy in (
            ("heretic_mpoa", "^", "#c46a1a", "Heretic MPOA (measured)", -105, -22),
            ("heretic_ara", "v", "#a8201a", "Heretic ARA (measured)", -46, 16),
        ):
            v = comparison[name]
            ax.scatter(max(v["kl_divergence"], 5e-4),
                       v["extended_refusal"]["refusal_rate"],
                       s=105, color=colour, marker=marker, zorder=4)
            ax.annotate(label,
                        (max(v["kl_divergence"], 5e-4),
                         v["extended_refusal"]["refusal_rate"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=8,
                        color=colour)

    # (q, gamma, x-offset, y-offset) for every measured redirect point.
    measured = [
        (0.95, 0.3, -12, 12),
        (0.95, 0.45, 14, -2),
        (0.95, 0.75, -14, -20),
        (0.5, 0.3, 10, -8),
        (0.25, 0.3, -30, 12),
    ]
    for q, gamma, dx, dy in measured:
        v = family[f"dense_q{q:g}_g{gamma:g}"]
        shipped = (q, gamma) == (0.95, 0.75)
        ax.scatter(v["kl_divergence"], v["heldout_300"]["refusal_rate"],
                   s=85, color="#7a4fa3", marker="o", zorder=4)
        if shipped:
            ax.scatter(v["kl_divergence"], v["heldout_300"]["refusal_rate"],
                       facecolors="none", edgecolors="#7a4fa3", s=230, lw=1.6,
                       zorder=6)
        ax.annotate(f"q={q:g}, $\\gamma$={gamma:g}",
                    (v["kl_divergence"], v["heldout_300"]["refusal_rate"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    color="#7a4fa3")

    dense = family["dense_q1_g0"]
    ax.scatter(dense["kl_divergence"], dense["heldout_300"]["refusal_rate"],
               s=105, color="#222222", marker="D", zorder=4)
    ax.annotate("dense removal (classic abliteration)",
                (dense["kl_divergence"], dense["heldout_300"]["refusal_rate"]),
                textcoords="offset points", xytext=(-24, 12), fontsize=8,
                color="#222222")

    ax.scatter(5e-4, base["refusal_rate"], s=105, color="#666666", marker="s",
               zorder=4)
    ax.annotate(f"base ({base['refusal_rate']:.2f})", (5e-4, base["refusal_rate"]),
                textcoords="offset points", xytext=(8, 4), fontsize=8,
                color="#666666")

    ax.set_xscale("log")
    ax.set_xticks([5e-4, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0])
    ax.set_xticklabels(["base", "0.01", "0.05", "0.10", "0.20", "0.50", "1.0", "5"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3.5e-4, 7)
    ax.set_ylim(-0.05, 0.75)
    ax.set_xlabel("KL divergence from base")
    ax.set_ylabel("refusal rate, 300 held-out prompts")
    ax.set_title("Qwen3.5-4B, plane 20, 425,984 atoms (ringed point is the shipped edit)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12-qwen35-frontier.png", dpi=160)
    plt.close(fig)

def tables(data: dict) -> None:
    lines = ["# Numbers behind the figures", ""]
    for key, d in data.items():
        lines += [f"## {d['cfg']['title']}", "",
                  "| model | report refusal | held-out refusal | KL | PPL |",
                  "| --- | --- | --- | --- | --- |"]
        for model in ("base", "taa", "shipped", "heretic_mpoa", "heretic_ara"):
            if model == "shipped":
                shipped = d["shipped"]
                lines.append(
                    f"| {LABELS['shipped']} | {shipped['report_rate']:.3f} "
                    f"({shipped['report_refusals']}/70) | {shipped['refusal_rate']:.3f} "
                    f"({shipped['refusals']}/{shipped['n']}) | {shipped['kl_divergence']:.4f} "
                    f"| {shipped['ppl']:.2f} |")
                continue
            r = d["table"][model]["report"]
            h = d["holdout"][model]
            lines.append(
                f"| {LABELS[model]} | {r['refusal_rate']:.3f} "
                f"({r['refusals']}/{r['n']}) | {h['refusal_rate']:.3f} "
                f"({h['refusals']}/{h['n']}) | {d['table'][model]['kl_divergence']:.4f} "
                f"| {d['table'][model]['perplexity']:.2f} |")
        lines += ["", "| model | arc_easy | piqa | winogrande |", "| --- | --- | --- | --- |"]
        for model in ("base", "taa", "shipped", "heretic_mpoa", "heretic_ara"):
            if model not in d["bench"]:
                continue
            b = d["bench"].get(model, {})
            cells = [f"{b[t]['acc,none']:.3f}" if t in b else "n/a"
                     for t in ("arc_easy", "piqa", "winogrande")]
            lines.append(f"| {LABELS[model]} | " + " | ".join(cells) + " |")
        lines.append("")

    sweep = load("redirect/gradient_summary.json")["results"]
    lines += ["## The redirect", "",
              "Qwen2.5-1.5B, plane 18, 35,043 atoms. Refusal is the "
              "100-prompt evaluation slice or the 300 held-out prompts.",
              "", "| target | gamma | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- |"]
    for key, label in [
        ("neg_r_g0.3", "-r"), ("neg_r_g0.5", "-r"), ("neg_r_g0.75", "-r"),
        ("neg_r_g1", "-r"),
        ("grad_g0.3", "gradient"), ("grad_g0.5", "gradient"),
        ("grad_g0.75", "gradient"), ("grad_g1", "gradient"),
        ("grad_orth_g0.3", "gradient, orthogonalised"),
        ("grad_orth_g0.5", "gradient, orthogonalised"),
        ("grad_orth_g0.75", "gradient, orthogonalised"),
        ("grad_orth_g1", "gradient, orthogonalised"),
    ]:
        r = sweep[key]
        lines.append(
            f"| {label} | {r['gamma']} | {r['refusal_100']['refusal_rate']:.3f} "
            f"| {r['heldout_300']['refusal_rate']:.3f} "
            f"| {r['kl_divergence']:.4f} | {r['perplexity_wikitext2']:.2f} |")
    lines.append("")

    configs = load("redirect/kl_split.json")["configs"]
    lines += ["## Where the KL goes", "",
              "| edit | KL | intended | elsewhere | intended share |",
              "| --- | --- | --- | --- | --- |"]
    for key, label in [
        ("removal_90", "removal, 90%"),
        ("neg_r_g0.5", "-r, gamma 0.5"),
        ("neg_r_g1", "-r, gamma 1.0"),
        ("grad_g0.3", "gradient, gamma 0.3"),
        ("grad_g0.5", "gradient, gamma 0.5"),
        ("grad_g0.75", "gradient, gamma 0.75"),
        ("dense_removal", "dense removal"),
    ]:
        c = configs[key]
        lines.append(
            f"| {label} | {c['kl_total']:.4f} "
            f"| {c['kl_in_accept_refuse_set']:.4f} "
            f"| {c['kl_outside_set']:.4f} "
            f"| {c['in_set_share'] * 100:.1f}% |")
    lines.append("")

    lines += ["## Category control", "",
              "| atoms | strength | cyber refusal | other refusal | KL |",
              "| --- | --- | --- | --- | --- |"]
    for label, name in [("5,000", "category_cyber_5k.json"),
                        ("20,000", "category_cyber_20k.json"),
                        ("50,000", "category_cyber_50k.json")]:
        d = load(f"redirect/{name}")
        lines.append(
            f"| {label} | {d['strength']:.1f} "
            f"| {d['eval_cat']['refusal_rate']:.3f} "
            f"| {d['eval_other']['refusal_rate']:.3f} "
            f"| {d['all']['kl_divergence']:.3f} |")
    lines.append("")

    lines += ["## Hybrid (efficiency-ranked)", "",
              "The bottom q of atoms by refusal load per unit of first-order "
              "harmless write is removed; the rest redirected.",
              "", "| gamma | q | removal atoms | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    fine = load("redirect/hybrid_efficiency_fine.json")["results"]
    coarse = load("redirect/hybrid_sweep.json")["results"]
    for gamma in (0.3, 0.75):
        rows = [(0.0, coarse[f"efficiency_q0_g{gamma:g}"])] + sorted(
            (v["quantile"], v) for v in fine.values()
            if abs(v["gamma"] - gamma) < 1e-9
        )
        for q, v in rows:
            lines.append(
                f"| {gamma:g} | {q:g} | {v['removal_atoms']} "
                f"| {v['refusal_100']['refusal_rate']:.3f} "
                f"| {v['heldout_300']['refusal_rate']:.3f} "
                f"| {v['kl_divergence']:.4f} "
                f"| {v['perplexity_wikitext2']:.2f} |"
            )
    lines.append("")

    lines += ["## Continuous per-atom gamma", "",
              "Every selected atom gets gamma * t(rank); the binary hybrid is the step.",
              "", "| gamma | mapping | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- |"]
    continuous = load("redirect/continuous_sweep.json")["results"]
    for value in continuous.values():
        lines.append(
            f"| {value['gamma']:g} | {value['mapping']} "
            f"| {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |"
        )
    lines.append("")

    lines += ["## The redirect on Qwen2.5-0.5B-Instruct", "",
              "Plane 14, 16,894 atoms. The target is fitted on this model's own prompts.",
              "", "| gamma | q | removal atoms | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    second = load("redirect/redirect_05b.json")["results"]
    for value in sorted(second.values(), key=lambda v: (v["gamma"], v["quantile"])):
        lines.append(
            f"| {value['gamma']:g} | {value['quantile']:g} "
            f"| {value['removal_atoms']} "
            f"| {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |"
        )
    lines.append("")

    family = load("redirect/dense_hybrid_family.json")["results"]
    lines += ["## Dense removal plus a redirect of the efficient atoms", "",
              "Qwen2.5-1.5B, plane 18, all 251,216 atoms. The atoms below the "
              "quantile q are removed; the rest carry the target at `gamma`. "
              "q=1 is dense removal.",
              "", "| gamma | q | removed | redirected | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for value in sorted(family.values(),
                        key=lambda v: (v["gamma"], v["quantile"])):
        lines.append(
            f"| {value['gamma']:g} | {value['quantile']:g} "
            f"| {value['removal_atoms']:,} "
            f"| {value['redirect_atoms']:,} "
            f"| {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |"
        )
    lines.append("")

    second = load("redirect/dense_hybrid_05b_family.json")["results"]
    lines += ["## The same family on Qwen2.5-0.5B-Instruct", "",
              "Plane 14, 117,072 atoms.",
              "", "| gamma | q | redirected | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for value in sorted(second.values(),
                        key=lambda v: (v["gamma"], v["quantile"])):
        lines.append(
            f"| {value['gamma']:g} | {value['quantile']:g} "
            f"| {value['redirect_atoms']:,} "
            f"| {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |"
        )
    lines.append("")

    rank = load("redirect/ranking.json")["results"]
    lines += ["## Which atoms get the target", "",
              "q=0.95, gamma=0.45. The same atom count is chosen three ways.",
              "", "| ranking | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- |"]
    for key, label in (("dense_q0.95_g0.45", "load per unit harmless write"),
                       ("load_q0.95_g0.45", "load"),
                       ("damage_q0.95_g0.45", "harmless write")):
        value = rank[key]
        lines.append(
            f"| {label} | {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |")
    lines.append("")

    smart = load("redirect/smart.json")["results"]
    lines += ["## Which direction the target points", "",
              "q=0.95, gamma=0.45, all atoms. The target is the acceptance "
              "gradient, modified as named.",
              "", "| target | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- |"]
    for key, label in (("grad_q0.95_g0.45", "acceptance gradient"),
                       ("grad_orth_q0.95_g0.45", "orthogonal to r"),
                       ("grad_good_q0.95_g0.45", "orthogonal to the mean harmless gradient"),
                       ("grad_pca_q0.95_g0.45", "orthogonal to the top 32 harmless directions"),
                       ("efficiency_q0.95_g0.45", "per-layer gamma by measured conversion")):
        value = smart[key]
        lines.append(
            f"| {label} | {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['perplexity_wikitext2']:.2f} |")
    lines.append("")

    batch = load("batch_timing.json")["batches"]
    lines += ["## Batch size", "",
              "Evaluation cost and result at four batch sizes, base model. The "
              "KL column is measured against a baseline taken at batch 32, so "
              "the nonzero values at larger batches are the size of the "
              "protocol difference, not a change in the model.",
              "", "| batch | generate | first-token logits | measure | refusal, 100 | KL |",
              "| --- | --- | --- | --- | --- | --- |"]
    for size in ("32", "64", "96", "128"):
        value = batch[size]
        median = lambda xs: sorted(xs)[len(xs) // 2]
        lines.append(
            f"| {size} | {median(value['refusal_seconds']):.2f} s "
            f"| {median(value['logits_seconds']):.2f} s "
            f"| {median(value['measure_seconds']):.2f} s "
            f"| {value['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} |")
    lines.append("")

    parity = load("batch_parity_1.5b.json")["results"]
    lines += ["## Protocol parity", "",
              "Every comparison checkpoint at both batch sizes, same harness.",
              "", "| model | batch | refusal, 100 | held out, 300 | KL | PPL |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in ("base", "dense_removal", "heretic_mpoa", "heretic_ara"):
        for size in ("32", "128"):
            value = parity[name][size]
            lines.append(
                f"| {LABELS[name]} | {size} "
                f"| {value['refusal_100']:.3f} "
                f"| {value['heldout_300']:.3f} "
                f"| {value['kl_divergence']:.4f} "
                f"| {value['perplexity_wikitext2']:.2f} |")
    lines.append("")

    lines += ["## Phrase weight", "",
              "lambda = 0 is the acceptance gradient; raising lambda raises the "
              "phrase but never surfaces it.",
              "", "| gamma | lambda | refusal, 100 | held out, 300 | KL | phrase logprob | starts | contains |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    weight = load("redirect/phrase_weight.json")["results"]
    for value in sorted(weight.values(), key=lambda v: (v["gamma"], v["lambda"])):
        lines.append(
            f"| {value['gamma']:g} | {value['lambda']:g} "
            f"| {value['refusal_100']['refusal_rate']:.3f} "
            f"| {value['heldout_300']['refusal_rate']:.3f} "
            f"| {value['kl_divergence']:.4f} "
            f"| {value['phrase_logprob']:.2f} "
            f"| {value['phrase_start_rate'] * 100:.0f}/100 "
            f"| {value['phrase_anywhere_rate'] * 100:.0f}/100 |"
        )
    lines.append("")

    t = load("timing.json")
    lines += ["## Time", "", "| stage | seconds |", "| --- | --- |"]
    for name, value in t["stages"].items():
        lines.append(f"| {name} | {value:.2f} |")
    lines += [f"| Heretic MPOA, 40 trials, whole process | "
              f"{t['heretic_clean_wall_seconds']['mpoa_40_trials']:.1f} |",
              f"| Heretic ARA, 40 trials, whole process | "
              f"{t['heretic_clean_wall_seconds']['ara_40_trials']:.1f} |", ""]
    (FIGURES / "numbers.md").write_text("\n".join(lines))


def qwen35_numbers() -> None:
    """Append the Qwen3.5-4B section to numbers.md when its files are present."""
    if not (RESULTS / "qwen35_base.json").exists():
        return
    base = load("qwen35_base.json")
    family = load("redirect/dense_hybrid_35b.json")["results"]
    export = load("redirect/export_35b.json")
    hold = load("holdout_Qwen3.5-4B.json")
    removal = family["dense_q1_g0"]
    shipped = family["dense_q0.95_g0.75"]
    lines = ["", "## Qwen3.5-4B", "",
             f"{base['n_layers']} layers, hidden {base['hidden_size']}, "
             f"attention atoms '{base['attention_atoms']}' "
             f"({base['atoms_per_layer']} per layer, layer types "
             f"{base['layer_types']}), plane 20 of {base['n_layers']}.",
             "",
             "| model | eval, 100 | held out, 300 | KL | PPL |",
             "| --- | --- | --- | --- | --- |",
             f"| base | {base['base']['refusal']['refusals']}/"
             f"{base['base']['refusal']['n']} | "
             f"{base['base_heldout_300']['refusals']}/"
             f"{base['base_heldout_300']['n']} | 0 | "
             f"{base['base']['perplexity_wikitext2']:.2f} |",
             f"| dense removal | {removal['refusal_100']['refusals']}/"
             f"{removal['refusal_100']['n']} | "
             f"{removal['heldout_300']['refusals']}/"
             f"{removal['heldout_300']['n']} | "
             f"{removal['kl_divergence']:.4f} | "
             f"{removal['perplexity_wikitext2']:.2f} |",
             f"| target on the top 5%, gamma=0.75 | "
             f"{shipped['refusal_100']['refusals']}/{shipped['refusal_100']['n']} | "
             f"{shipped['heldout_300']['refusals']}/{shipped['heldout_300']['n']} | "
             f"{shipped['kl_divergence']:.4f} | "
             f"{shipped['perplexity_wikitext2']:.2f} |"]
    if (RESULTS / "comparison_Qwen3.5-4B.json").exists():
        comparison = load("comparison_Qwen3.5-4B.json")
        for name, label in (("heretic_mpoa", "Heretic MPOA, 40 trials"),
                            ("heretic_ara", "Heretic ARA, 40 trials")):
            v = comparison[name]
            ev, held = v["refusal"], v["extended_refusal"]
            lines.append(
                f"| {label} | {ev['refusals']}/{ev['n']} | "
                f"{held['refusals']}/{held['n']} | "
                f"{v['kl_divergence']:.4f} | "
                f"{v['perplexity_wikitext2']:.2f} |")
    lines += [
             "",
             f"Exported checkpoint after reload: {export['refusal_100']['refusals']}/100, "
             f"{export['heldout_300']['refusals']}/300, KL {export['kl_divergence']:.4f}, "
             f"PPL {export['perplexity_wikitext2']:.2f}.",
             "Heretic PR #211 reports for this model: MPOA still above 50/100 "
             "after abliteration; ARA 4/100 at 0.1396 KL (reported, not "
             "re-measured).", ""]
    with (FIGURES / "numbers.md").open("a") as f:
        f.write("\n".join(lines))


def main() -> None:
    global RESULTS, FIGURES
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT),
                    help="directory with tables_main.json, holdout_*.json, "
                         "bench_*.json, timing.json and redirect/ and graph/")
    ap.add_argument("--output", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()
    RESULTS = Path(args.results)
    FIGURES = Path(args.output)
    FIGURES.mkdir(parents=True, exist_ok=True)

    data = gather()
    tradeoff(data)
    refusal_bars(data)
    capability(data)
    timing()
    redirect_pareto()
    redirect_kl_split()
    redirect_category()
    graph_view()
    hybrid_sweep()
    qwen35_frontier()
    hybrid_05b()
    frontier()
    tables(data)
    qwen35_numbers()
    print(f"wrote {FIGURES}")


if __name__ == "__main__":
    main()
