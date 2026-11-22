"""Check every number in the README against the result files.

The README quotes results in six tables and recomputes about twenty ratios in
prose.  A number that drifts from the file it cites is the failure mode this
catches, so the check reads both sides: the table cell or the sentence as text,
and the JSON field it came from.  Any mismatch prints both values and the script
exits non-zero.

    PYTHONPATH=. .venv/bin/python scripts/46_check_readme.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results"




def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def tables(text: str) -> list[tuple[tuple[str, ...], list[list[str]]]]:
    """Markdown tables in document order, as ``(header, rows)``.

    A list rather than a dict: two sections carry tables with the same header
    (capability, one per model), and both must be checked.
    """
    found: list[tuple[tuple[str, ...], list[list[str]]]] = []
    current: list[list[str]] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            current = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue  # the header separator, not a row
        if current is None:
            current = []
            found.append((tuple(cells), current))
        else:
            current.append(cells)
    return found


def clean(cell: str) -> str:
    return cell.replace("*", "").replace("`", "").strip()


def number(cell: str) -> float:
    return float(clean(cell))


def sci(x: float, digits: int = 1) -> str:
    """Format 0.00048 as ``4.8e-4``, which is how the README writes it."""
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


FAILURES: list[str] = []


def check(where: str, got: str, want: float, _kind: str = "") -> None:
    """The cell may print ``want`` at its own precision and no more."""
    text = clean(got)
    decimals = len(text.split(".")[1]) if "." in text else 0
    slack = 0.5 * 10 ** (-decimals) + 1e-9
    if abs(number(got) - want) > slack:
        FAILURES.append(f"{where}: readme {got} vs data {want:.6f}")


def main() -> None:
    text = README.read_text()
    found = tables(text)

    main_table = load("tables_main.json")
    holdout = load("holdout_Qwen2.5-1.5B-Instruct.json")
    bench = {name: load(f"bench_{name}.json")["tasks"]
             for name in ("base", "taa", "taa_target", "heretic_mpoa", "heretic_ara")}

    def rows_for(header, index=0):
        matches = [rows for head, rows in found if head == header]
        return matches[index]

    # 1. The headline table.
    header = ("edit", "refusal, 70", "refusal, 300 held out", "KL", "PPL", "arc_easy")
    rows = {r[0].replace("*", ""): r for r in rows_for(header)}
    for label, key, bench_key in (
        ("base", "base", "base"),
        ("dense removal", "taa", "taa"),
        ("+ target on the top 5%", "shipped", "taa_target"),
        ("Heretic MPOA, 40 trials", "heretic_mpoa", "heretic_mpoa"),
        ("Heretic ARA, 40 trials", "heretic_ara", "heretic_ara"),
    ):
        row = rows[label]
        if key == "shipped":
            point = load("redirect/dense_hybrid_family.json")["results"]["dense_q0.95_g0.45"]
            responses = load("redirect/dense_hybrid_responses.json")[
                "results"]["dense_q0.95_g0.45"]["responses"][30:]
            report = sum(r["refusal"] for r in responses) / len(responses)
            held = point["heldout_300"]["refusal_rate"]
            kl = point["kl_divergence"]
            ppl = point["perplexity_wikitext2"]
        else:
            report = main_table[key]["report"]["refusal_rate"]
            held = holdout[key]["refusal_rate"]
            kl = main_table[key]["kl_divergence"]
            ppl = main_table[key]["perplexity"]
        check(f"headline {label} report", row[1], report, "rate")
        check(f"headline {label} held", row[2], held, "rate")
        check(f"headline {label} kl", row[3], kl, "kl")
        check(f"headline {label} ppl", row[4], ppl, "ppl")
        check(f"headline {label} arc", row[5],
              bench[bench_key]["arc_easy"]["acc,none"], "acc")

    # 2. The family table.
    family = load("redirect/dense_hybrid_family.json")["results"]
    header = ("target on", "atoms", "refusal, 100", "held out, 300", "KL", "PPL")
    for row in rows_for(header):
        label = row[0].replace("*", "")
        atoms = int(row[1].replace(",", "").replace("*", ""))
        match = re.search(r"gamma = ([0-9.]+)", label)
        if not match:
            point = family["dense_q1_g0"]
        else:
            gamma = float(match.group(1))
            candidates = [v for v in family.values()
                          if v["redirect_atoms"] == atoms
                          and abs(v["gamma"] - gamma) < 1e-9]
            assert len(candidates) == 1, (label, atoms, len(candidates))
            point = candidates[0]
        check(f"family {label} refusal", row[2],
              point["refusal_100"]["refusal_rate"], "rate")
        check(f"family {label} held", row[3],
              point["heldout_300"]["refusal_rate"], "rate")
        check(f"family {label} kl", row[4], point["kl_divergence"], "kl")
        check(f"family {label} ppl", row[5], point["perplexity_wikitext2"], "ppl")

    # 3. The ranking table.
    ranking = load("redirect/ranking.json")["results"]
    header = ("the same 12,561 target atoms, chosen by",
              "refusal, 100", "held out, 300", "KL", "PPL")
    keys = {"load per unit harmless write": "dense_q0.95_g0.45",
            "load": "load_q0.95_g0.45", "the largest harmless write": "damage_q0.95_g0.45"}
    for row in rows_for(header):
        point = ranking[keys[row[0]]]
        check(f"ranking {row[0]} refusal", row[1],
              point["refusal_100"]["refusal_rate"], "rate")
        check(f"ranking {row[0]} held", row[2],
              point["heldout_300"]["refusal_rate"], "rate")
        check(f"ranking {row[0]} kl", row[3], point["kl_divergence"], "kl")
        check(f"ranking {row[0]} ppl", row[4], point["perplexity_wikitext2"], "ppl")

    # 4. The target variants.
    smart = load("redirect/smart.json")["results"]
    header = ("target", "refusal, 100", "held out, 300", "KL", "PPL")
    keys = {"the acceptance gradient": "grad_q0.95_g0.45",
            "orthogonal to `r`": "grad_orth_q0.95_g0.45",
            "orthogonal to the mean harmless top-1 gradient": "grad_good_q0.95_g0.45",
            "orthogonal to the top 32 harmless top-1 gradient directions": "grad_pca_q0.95_g0.45",
            "per-layer `gamma` from measured conversion per unit KL": "efficiency_q0.95_g0.45"}
    for row in rows_for(header):
        point = smart[keys[row[0]]]
        check(f"variant {row[0]} refusal", row[1],
              point["refusal_100"]["refusal_rate"], "rate")
        check(f"variant {row[0]} held", row[2],
              point["heldout_300"]["refusal_rate"], "rate")
        check(f"variant {row[0]} kl", row[3], point["kl_divergence"], "kl")
        check(f"variant {row[0]} ppl", row[4], point["perplexity_wikitext2"], "ppl")

    # 5. The sparse redirect.
    sweep = load("redirect/gradient_summary.json")["results"]
    probe = load("redirect/pilot_probe.json")
    header = ("edit", "refusal, 100", "refusal, 300 held out", "KL", "PPL")
    keys = {"removal, same atoms": ("probe", "removal_s1"),
            "redirect to `-r`, `gamma = 1`": ("sweep", "neg_r_g1"),
            "redirect to the gradient, `gamma = 0.3`": ("sweep", "grad_g0.3"),
            "redirect to the gradient, `gamma = 0.75`": ("sweep", "grad_g0.75"),
            "dense removal, 251k atoms": ("probe", "dense_removal")}
    for row in rows_for(header):
        source, key = keys[row[0]]
        point = probe[key] if source == "probe" else sweep[key]
        check(f"sparse {row[0]} refusal", row[1],
              point["refusal_100"]["refusal_rate"], "rate")
        check(f"sparse {row[0]} held", row[2],
              point["heldout_300"]["refusal_rate"], "rate")
        check(f"sparse {row[0]} kl", row[3], point["kl_divergence"], "kl")
        check(f"sparse {row[0]} ppl", row[4], point["perplexity_wikitext2"], "ppl")

    # 6. Capability, one table per model.
    header = ("model", "arc_easy", "piqa", "winogrande")
    labels = ("base", "dense removal", "+ target on the top 5%",
              "Heretic MPOA", "Heretic ARA")
    for index, model, names in (
        (0, "1.5B", ("base", "taa", "taa_target", "heretic_mpoa", "heretic_ara")),
        (1, "0.5B", ("base_05b", "taa_05b", "taa_target_05b",
                     "heretic_mpoa_05b", "heretic_ara_05b")),
    ):
        rows = {r[0].replace("*", ""): r for r in rows_for(header, index)}
        for label, name in zip(labels, names):
            scores = load(f"bench_{name}.json")["tasks"]
            row = rows[label]
            for column, task in zip(row[1:], ("arc_easy", "piqa", "winogrande")):
                check(f"capability {model} {label} {task}", column,
                      scores[task]["acc,none"], "acc")

    # 7. The derived numbers in the prose.  Each value is recomputed from the
    #    file it cites, and the sentence must carry it at the stated precision.
    flat = " ".join(text.split())

    def expect(label: str, phrase: str, count: int = 1) -> None:
        """The phrase must appear at least ``count`` times."""
        if flat.count(phrase) < count:
            FAILURES.append(f"prose {label}: {phrase!r} found "
                            f"{flat.count(phrase)} times, want {count}")

    dense = family["dense_q1_g0"]
    shipped = family["dense_q0.95_g0.45"]
    dense_rate = holdout["taa"]["refusal_rate"]
    mpoa_rate = holdout["heretic_mpoa"]["refusal_rate"]
    expect("dense held", f"{dense_rate:.3f} for dense removal")
    expect("mpoa held", f"{mpoa_rate:.3f} for Heretic MPOA")

    # The chosen point against each Heretic mode and against removal.
    expect("mpoa refusal ratio",
           f"{mpoa_rate / shipped['heldout_300']['refusal_rate']:.1f} times worse")
    expect("mpoa kl ratio",
           f"{shipped['kl_divergence'] / main_table['heretic_mpoa']['kl_divergence']:.1f} times cheaper")
    expect("ara kl ratio",
           f"{main_table['heretic_ara']['kl_divergence'] / shipped['kl_divergence']:.1f} times")
    expect("family kl ratio",
           f"{shipped['kl_divergence'] / dense['kl_divergence']:.2f} times the KL")

    # The hybrid's KL cut, against redirecting every atom.
    fine = load("redirect/hybrid_efficiency_fine.json")["results"]
    for key, base in (("efficiency_q0.25_g0.3", "grad_g0.3"),
                      ("efficiency_q0.3_g0.75", "grad_g0.75")):
        cut = 100 * (1 - fine[key]["kl_divergence"] / sweep[base]["kl_divergence"])
        expect(f"hybrid cut {key}", f"{cut:.1f}% lower")

    # The batch-size saving, from the medians the harness records.
    batches = load("batch_timing.json")["batches"]
    m32, m128 = batches["32"]["measure_median"], batches["128"]["measure_median"]
    expect("batch 32", f"{m32:.1f} s")
    expect("batch 128", f"{m128:.1f} s")
    expect("batch speedup", f"{m32 / m128:.2f} times faster")
    expect("batch saving", f"{100 * (1 - m128 / m32):.0f}% cheaper")
    parity = load("batch_parity_1.5b.json")["results"]
    worst_kl = max(abs(v["32"]["kl_divergence"] - v["128"]["kl_divergence"])
                   for v in parity.values())
    expect("batch kl shift", f"{sci(worst_kl, 0)} KL shift")

    # The attribution gate and its positive-load split.
    identity = load("sanity.json")["identity"]
    worst = max(identity["max_relative_error_atom_sum_vs_writes"],
                identity["max_relative_error_residual_vs_sum"])
    expect("identity error", f"{sci(worst)} relative", count=2)
    paths = load("analysis/paths_layers.json")
    total = paths["both"]["predicted_load_removed"]
    attn = total - paths["ffn_only"]["predicted_load_removed"]
    expect("attn share", f"{100 * attn / total:.0f}% attention")
    expect("signed sum", f"{sum(identity['per_layer_attributed']):.1f}")

    # The continuous and phrase negatives.
    cont = load("redirect/continuous_sweep.json")["results"]
    square = cont["sqrt_g0.75"]
    expect("continuous tie", f"{square['heldout_300']['refusal_rate']:.3f} held out and KL "
                             f"{square['kl_divergence']:.3f}")
    binary = fine["efficiency_q0.3_g0.75"]
    expect("continuous binary", f"{binary['heldout_300']['refusal_rate']:.3f} at "
                                f"{binary['kl_divergence']:.3f}")
    phrase = load("redirect/phrase_weight.json")["results"]
    expect("phrase 0.75",
           f"{phrase['lambda0_g0.75']['phrase_logprob']:.1f} to "
           f"{phrase['lambda4_g0.75']['phrase_logprob']:.1f}")
    expect("phrase 0.3",
           f"{phrase['lambda0_g0.3']['phrase_logprob']:.1f} to "
           f"{phrase['lambda4_g0.3']['phrase_logprob']:.1f}")

    # 8. The Qwen3.5-4B section.  The two Heretic rows are quoted from PR #211
    #    and carry no file, so only the measured rows and the prose are checked.
    base35 = load("qwen35_base.json")
    family35 = load("redirect/dense_hybrid_35b.json")["results"]
    family35_meta = load("redirect/dense_hybrid_35b.json")
    export35 = load("redirect/export_35b.json")
    hold35 = load("holdout_Qwen3.5-4B.json")
    bench35 = {k: load(f"bench_{k}_35b.json")["tasks"] for k in ("base", "taa")}

    header = ("edit", "refusal, 100 eval", "refusal, 300 held out", "KL", "PPL")
    rows = {r[0].replace("*", ""): r for r in rows_for(header)}
    base_held35 = hold35["base"]["refusal_rate"]
    removal35 = family35["dense_q1_g0"]
    shipped35 = family35["dense_q0.95_g0.75"]
    q35_rows = (
        ("base", base35["base"]["refusal"]["refusal_rate"], base_held35,
         0.0, base35["base"]["perplexity_wikitext2"]),
        ("dense removal", removal35["refusal_100"]["refusal_rate"],
         removal35["heldout_300"]["refusal_rate"], removal35["kl_divergence"],
         removal35["perplexity_wikitext2"]),
        ("+ target on the top 5%", shipped35["refusal_100"]["refusal_rate"],
         shipped35["heldout_300"]["refusal_rate"], shipped35["kl_divergence"],
         shipped35["perplexity_wikitext2"]),
    )
    if (RESULTS / "comparison_Qwen3.5-4B.json").exists():
        comp35 = load("comparison_Qwen3.5-4B.json")
        q35_rows += (
            ("Heretic MPOA, 40 trials", comp35["heretic_mpoa"]["refusal"]["refusal_rate"],
             comp35["heretic_mpoa"]["extended_refusal"]["refusal_rate"],
             comp35["heretic_mpoa"]["kl_divergence"],
             comp35["heretic_mpoa"]["perplexity_wikitext2"]),
            ("Heretic ARA, 40 trials", comp35["heretic_ara"]["refusal"]["refusal_rate"],
             comp35["heretic_ara"]["extended_refusal"]["refusal_rate"],
             comp35["heretic_ara"]["kl_divergence"],
             comp35["heretic_ara"]["perplexity_wikitext2"]),
        )
    for label, eval_r, held_r, kl, ppl in q35_rows:
        row = rows[label]
        check(f"q35 {label} eval", row[1], eval_r)
        check(f"q35 {label} held", row[2], held_r)
        check(f"q35 {label} kl", row[3], kl)
        check(f"q35 {label} ppl", row[4], ppl)

    # The exported checkpoint must reproduce the in-memory family point.
    if abs(export35["refusal_100"]["refusal_rate"]
           - shipped35["refusal_100"]["refusal_rate"]) > 1e-9:
        FAILURES.append("q35 export refusal differs from the family point")
    if abs(export35["heldout_300"]["refusal_rate"]
           - shipped35["heldout_300"]["refusal_rate"]) > 1e-9:
        FAILURES.append("q35 export holdout differs from the family point")
    if abs(export35["kl_divergence"] - shipped35["kl_divergence"]) > 5e-4:
        FAILURES.append("q35 export KL differs from the family point")

    expect("q35 removal share",
           f"{100 * (base_held35 - removal35['heldout_300']['refusal_rate']) / base_held35:.0f}% of the gap")
    expect("q35 shipped share",
           f"{100 * (base_held35 - shipped35['heldout_300']['refusal_rate']) / base_held35:.0f}% at "
           f"{shipped35['kl_divergence'] / removal35['kl_divergence']:.1f} times removal's KL")
    expect("q35 half atoms",
           f"{family35['dense_q0.5_g0.3']['kl_divergence'] / shipped35['kl_divergence']:.0f} times that KL")
    for task in ("arc_easy", "piqa", "winogrande"):
        expect(f"q35 capability {task}",
               f"{bench35['base'][task]['acc,none']:.3f} to "
               f"{bench35['taa'][task]['acc,none']:.3f}")

    calib35 = load("analysis/qwen35_calibration.json")
    expect("q35 plane and atoms",
           f"Plane {calib35['chosen_plane']}, {family35_meta['atoms']:,} atoms")
    expect("q35 hybrid layer count",
           f"{base35['layer_types']['linear_attention']} of its "
           f"{base35['n_layers']} layers are linear attention")
    expect("q35 frontier sentence",
           f"from {base_held35:.2f} to "
           f"{shipped35['heldout_300']['refusal_rate']:.2f} — below measured "
           f"MPOA's {comp35['heretic_mpoa']['extended_refusal']['refusal_rate']:.2f} — at "
           f"{shipped35['kl_divergence'] / comp35['heretic_mpoa']['kl_divergence']:.1f} "
           f"times MPOA's KL")
    # The Heretic prose numbers: the PR citation and the measured run's ratios.
    # Present on snapshots that carry the measured Heretic runs.
    if (RESULTS / "comparison_Qwen3.5-4B.json").exists():
        trials35 = load("heretic_35b_trials.json")
        bench35_heretic = {k: load(f"bench_{k}_35b.json")["tasks"]
                           for k in ("heretic_mpoa", "heretic_ara")}
        expect("q35 pr mpoa claim", "more than 50 of 100 harmful prompts")
        expect("q35 pr ara claim", "4 of 100 at 0.1396 KL")
        expect("q35 heretic optimiser trials",
               f"{trials35['mpoa']['refusals']}/100 for its best MPOA trial")
        expect("q35 heretic optimiser ara",
               f"0/100 at {trials35['ara']['kl_divergence']:.4f} for its best ARA trial")
        expect("q35 mpoa kl ratio",
               f"{shipped35['kl_divergence'] / comp35['heretic_mpoa']['kl_divergence']:.1f} "
               f"times MPOA's KL")
        expect("q35 ara kl ratio",
               f"{comp35['heretic_ara']['kl_divergence'] / shipped35['kl_divergence']:.1f} "
               f"times the shipped point's KL")
        expect("q35 ara ppl", f"{comp35['heretic_ara']['perplexity_wikitext2']:.2f} against "
                              f"{base35['base']['perplexity_wikitext2']:.2f}")
        expect("q35 mpoa capability",
               ", ".join(f"{bench35_heretic['heretic_mpoa'][t]['acc,none']:.3f}"
                         for t in ("arc_easy", "piqa", "winogrande")))
        expect("q35 ara capability",
               ", ".join(f"{bench35_heretic['heretic_ara'][t]['acc,none']:.3f}"
                         for t in ("arc_easy", "piqa", "winogrande")))

    if FAILURES:
        print(f"{len(FAILURES)} mismatches")
        for line in FAILURES:
            print("  " + line)
        return 1

    print("README tables and prose agree with the result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
