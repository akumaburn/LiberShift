# Frozen results

These files are the evidence behind the numbers in the README and
`docs/results.md`. They were copied from a working directory (`out/`) after the
runs finished, and they are committed so the claims can be checked without a
GPU.

The scripts write new results to `out/`, which is not tracked. To point the
figure script at this snapshot:

```sh
PYTHONPATH=. .venv/bin/python scripts/18_figures.py --results results --output docs/figures
```

| path | contents |
| --- | --- |
| `tables_main.json`, `tables__05b.json` | report-split refusal, KL and perplexity per model |
| `holdout_Qwen2.5-*.json` | held-out refusal per model, 300 prompts from JailbreakBench and BeaverTails |
| `holdout_Qwen3.5-4B.json` | the same for Qwen3.5-4B: base 180/300, shipped edit 69/300, Heretic MPOA 88/300, Heretic ARA 3/300 |
| `comparison_Qwen3.5-4B.json` | the four Qwen3.5-4B checkpoints measured by one harness: report refusal, KL, perplexity and the extended holdout per checkpoint |
| `eval/Qwen3.5-4B_*.json` | the same per checkpoint, with every response |
| `heretic_35b_trials.json` | Heretic's own per-trial scores on Qwen3.5-4B, parsed from the run logs; best trials MPOA 25/100 at 0.0262 KL and ARA 0/100 at 0.5427 |
| `bench_*.json` | lm-eval accuracy on `arc_easy`, `piqa`, `winogrande`, 300 samples each |
| `eval/*.json` | per-model evaluation with the generated response for every prompt |
| `analysis/calibration.json` | the six-plane calibration and the two failing strength schedules |
| `analysis/calibration_05b.json` | the 0.5B calibration: planes 14, 16 and 18 tie at 1/30, chosen plane 14, held-out refusal; the 70-prompt report split is in `tables__05b.json` |
| `analysis/direction_geometry.json` | direction cosines, the uncentered and centered energy shares, and the teacher shift's cosine with `-r` per plane |
| `analysis/plane_sweep.json` | propagation scores for every plane, plus three evaluated planes |
| `analysis/planes/plane_*.json` | one evaluation per candidate plane, with the response to every prompt |
| `analysis/atoms/*.json` | atom-sparsity runs: 50, 90, 99% of load, the two-round 90% run, and the two dense controls |
| `analysis/holdout_by_plane_05b.json` | held-out refusal for planes 14, 16, 18 and 20 on the 0.5B model |
| `analysis/causal_layers.json` | single-layer causal curve, layer ranges, propagation scores |
| `analysis/paths_layers.json` | attention-only, FFN-only and layer-subset ablations |
| `analysis/schedules.json` | layer-schedule experiments |
| `analysis/strength_sweep.json` | over-ablation sweep, strength 1.0 to 1.45 |
| `redirect/pilot_plane18.json` | the redirect sweep: removal against sign flip at four load targets |
| `redirect/pilot_probe.json` | activation ratio, top-5 edge-list change (the `down_top_token_changed` field) and held-out refusal for redirect, removal and dense removal |
| `redirect/pilot_rotation_s2.json`, `redirect/pilot_removal_s1.json`, `redirect/pilot_dense_removal.json` | one evaluation per control in the probe |
| `redirect/heads_only.json`, `redirect/features_only.json` | head-only and feature-only redirects |
| `redirect/teacher_target.json` | redirect toward a compliant model's residual shift, three scales |
| `redirect/category_cyber_*.json` | category-selective redirect at three atom budgets |
| `redirect/gradient_*.json` | the acceptance-gradient sweep, one file per target and scale, with every response |
| `redirect/gradient_summary.json` | the same sweep without the responses |
| `redirect/gradient_timing.json` | the acceptance gradient's measured cost: 2.47 s for 400 prompts on the 5080 |
| `redirect/hybrid_sweep.json` | the coarse hybrid sweep: damage and efficiency rankings, both targets |
| `redirect/hybrid_efficiency_fine.json` | the efficiency-ranked hybrid at q=0.05 to 0.40, both targets |
| `redirect/dense_hybrid_family.json` | **the shipped family**: dense removal with the target on the efficient top `1-q`, 24 points at `gamma` 0.3 to 0.8, 251,216 atoms |
| `redirect/dense_hybrid_05b_family.json` | the same family on Qwen2.5-0.5B-Instruct, plane 14, 117,072 atoms |
| `redirect/dense_hybrid_35b.json` | the family on Qwen3.5-4B, plane 20, 425,984 atoms, six points; dense removal leaves 0.61/0.387 and the shipped point q=0.95, gamma=0.75 reaches 0.49/0.230 at 0.1944 KL |
| `redirect/dense_hybrid_responses.json` | the stored 100 evaluation responses at q=0.95 and q=0.995, for the report split |
| `redirect/dense_hybrid_05b_responses.json` | the same for the 0.5B model at q=0.95, `gamma` 0.6 |
| `redirect/dense_hybrid_samples.json` | six responses at the shipped point: substantive compliance |
| `redirect/ranking.json` | the same 12,561 target atoms chosen by efficiency, by load and by raw harmless write |
| `redirect/smart.json` | four target directions and the per-layer gamma allocation; none moves the frontier |
| `redirect/dense_hybrid.json`, `dense_hybrid_fine.json` | the earlier coarse sweeps at q<=0.5; superseded by the family file, kept as provenance |
| `redirect/continuous_sweep.json` | the continuous per-atom gamma variants; the binary step matches or beats every one |
| `redirect/redirect_05b.json` | the redirect and hybrid on Qwen2.5-0.5B-Instruct, plane 14, 16,894 atoms |
| `redirect/phrase_contrast.json` | the earlier phrase-only and phrase-plus-contrast run; its single phrase_accept point is not reproduced by the weight sweep |
| `redirect/phrase_accept_samples.json` | samples from the earlier unnormalised phrase+acceptance mix: acceptance language, not the literal phrase |
| `redirect/phrase_weight.json` | the lambda sweep: lambda=0 is best, and "I can do that" appears in 0/100 at every lambda |
| `qwen35_base.json` | Qwen3.5-4B baseline: refusal 0.93 on the evaluation slice, 0.60 held out, perplexity 10.72; hybrid architecture recorded (columns atoms) |
| `analysis/qwen35_calibration.json` | the Qwen3.5-4B plane calibration: dense removal on the 70-prompt slice for planes 16 to 24, chosen plane 20 |
| `redirect/export_35b.json` | the shipped Qwen3.5-4B export: q=0.95, gamma=0.75, plane 20, measured after the safetensors round trip (0.490 / 0.230 / 0.1944 KL / 10.87 PPL) |
| `bench_heretic_mpoa_35b.json`, `bench_heretic_ara_35b.json` | capability for the two measured Heretic checkpoints on Qwen3.5-4B |
| `redirect/kl_split.json` | the harmless-prompt KL split into acceptance/refusal mass and everything else |
| `redirect/export/redirect_g*.json`, `.npz` | the operator patch for each exported redirect: directions, targets, indices, scales |
| `redirect/export/redirect_g*_verify.json` | replay on a pristine checkpoint, and the reloaded-model measurements |
| `graph/graph_grad_g0.3.json` | the 200 highest-load features: target and target-list change, walk before against after |
| `graph/graph_grad_g1.json`, `graph/graph_neg_r_g1.json` | the same at `gamma = 1.0` with each target |
| `sanity.json` | trace contract and attribution identity checks |
| `parity.json` | an independent engine's f32 layer dump against transformers, plane by plane |
| `patch_verify.json` | 1.5B patch size, SHA-256 and maximum logit difference on replay |
| `patch_verify_05b.json` | the same for the 0.5B patch: 380,327 bytes, 117,072 atoms, 0.0 logit difference |
| `timing.json` | stage timings for the method and whole-process timings for Heretic |
| `edit_timing.json` | dense vs sparse vs hybrid edit and selection timings; inference is identical |
| `batch_timing.json` | evaluation cost at batch 32, 64, 96 and 128 |
| `batch_parity_1.5b.json` | base, dense removal and both Heretic modes at batch 32 and 128 |
| `bench_taa_target.json` | capability of the exported 1.5B shipped checkpoint: 0.720/0.757/0.650 |
| `bench_taa_target_05b.json` | capability of the exported 0.5B shipped checkpoint: 0.570/0.707/0.557 |

Refusal rates are Heretic's keyword metric. `eval/*.json` keeps the raw
responses so the metric itself can be audited or replaced.
