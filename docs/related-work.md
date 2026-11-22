# LiberShift and prior work

This document states what each part of LiberShift owes to earlier work, and
which claims survive as new. The README leads with the new claims; this file
draws the boundary.

The short version: the removal edit is Arditi et al.'s weight orthogonalization
with grimjim's norm preservation, and the comparison is Heretic's. The
contribution is the redirect: a new edit rule, the measured gain it produces at
the same atoms, and the measurements that rule out the cheaper variants. The
boundary is stated work by work below.

## The edit

### Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024)

[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

The paper introduces two interventions. The first ablates the refusal direction
from activations at inference. The second writes the same removal into the
weights: for every matrix `W_out` that writes to the residual stream,

```
W_out <- W_out - r_hat r_hat^T W_out
```

(§4.1, Eq. 5). It applies one direction, taken from a chosen layer, to every
matrix. It selects the direction by searching over source layer and token
position, filtering candidates by KL divergence and refusal score, and pruning
the last 20% of layers. It evaluates the orthogonalized models on HarmBench and
on MMLU, ARC, GSM8K and TruthfulQA.

LiberShift takes the edit, the single direction applied at every layer, and the
idea of choosing the source layer by measured refusal. The differences from
Arditi are the norm restoration (from grimjim), the reduction of the search to
six planes, the attribution, and the patch export. The capability evaluation is
the same kind of evaluation.

### grimjim (Jim Lai), *Projected Abliteration* and *Norm-Preserving Biprojected Abliteration* (2025)

[Projected abliteration](https://huggingface.co/blog/grimjim/projected-abliteration) ·
[norm-preserving biprojected](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration),
6 November 2025

Projected abliteration removes only the component of the refusal direction that
is mechanistically relevant, after projecting against the harmless mean.
Norm-preserving biprojected abliteration decomposes each row into magnitude and
direction, applies the rank-1 removal to the directional component, renormalises
each row, and recombines the original magnitudes. It applies the edit to
`self_attn.o_proj` and `mlp.down_proj`, states that the update remains rank one,
and selects layers with a heuristic that combines signal-to-noise ratio and
cosine dissimilarity.

LiberShift applies the same goal on a different axis. It orthogonalises each
atom's write vector, a column of `down_proj` or a block of columns of `o_proj`,
and rescales that vector to its original norm; grimjim rescales each row of the
matrix. The rank-1 form and the choice of `o_proj` and `down_proj` come from
here. The differences are that LiberShift chooses the plane by a calibration
split and reports held-out refusal, KL and capability, and that it stores the
edit as a replayable patch.

### Heretic

[github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)

Heretic is the comparison. It combines a modifier with a TPE optimiser over
modifier parameters. Its directional modifier supports projected abliteration
and norm preservation (MPOA); its default modifier, Arbitrary-Rank Ablation
(ARA), searches a higher-dimensional edit. The comparison in this repository
uses Heretic 2.2.0.dev0 at commit `64e9991`, 40 trials, 15 startup trials, seed
12345, and Heretic's refusal markers, KL definition, system prompt and greedy
decoding. The measurement harness is shared, so the comparison is like for like.

The difference is the search. Heretic optimises many parameters over many full
evaluations. LiberShift fixes every parameter except the plane, and calibrates
that one on 30 prompts. On the 1.5B model the fixed-cost version reaches a better
held-out point at the same KL; on the 0.5B model the search wins on damage. The
result is model-dependent and is reported that way.

The ARA pull request ([#211](https://github.com/p-e-w/heretic/pull/211)) is also
the public record of a failure mode for the removal family. On Qwen3.5-4B,
directional ablation leaves more than 50 of 100 harmful prompts refused, and
the ARA search brings that to 4 of 100 at 0.1396 KL; the thread describes Qwen
as a family that resists the classic edit. That report is what sent this study
to Qwen3.5-4B: the model is a modern hybrid-attention decoder on which rank-one
removal is known to fall short, so it tests whether the redirect still buys
anything where removal does not.

### FailSpy, `abliterator`, and the community implementations

[github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator)

The common abliteration implementation removes the refusal direction from
`o_proj` and `down_proj`. It does not restore row norms and does not calibrate a
plane. It establishes the target matrices that LiberShift also uses.

### Activation steering, concept erasure, and re-pointing a write

The redirect's nearest neighbours work on a different object or with a different
target, so the boundary is short and worth stating exactly.

*Activation addition* (Turner et al., [arXiv:2308.10248](https://arxiv.org/abs/2308.10248))
adds a direction to the residual stream at inference. The redirect adds nothing
at inference. It changes the weights that write into the stream, so the shift is
permanent and the exported model runs unmodified.

*Concept erasure* (LEACE, Belrose et al.,
[arXiv:2306.03819](https://arxiv.org/abs/2306.03819); INLP, Ravfogel et al.,
[arXiv:2004.06638](https://arxiv.org/abs/2004.06638)) removes a concept from
activations while preserving the rest of the representation. The redirect removes
the concept's component from a weight and writes a measured replacement, and it
acts on weights rather than on activations.

Abliteration (Arditi et al.) deletes the component. It is the `gamma = 0` case of
the redirect. I have not found the `gamma > 0` form with a measured target in the
literature cited here. The nearest statement of the idea is the steering-vector
result that a direction added to the residual stream changes behaviour (Turner et
al.; Zou et al., [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)). The
redirect applies that observation to the weight that writes the direction, and
adds the sparse atom selection and the gradient target.

Read the claim narrowly. The update is one line, and the contribution is the
measurement of what it does to refusal, KL and held-out prompts.

## Attribution and circuits

### RDSHA, *Safety Alignment Should Be Made More Than Just A Few Attention Heads* (2025)

[arXiv:2508.19697](https://arxiv.org/abs/2508.19697)

RDSHA uses the model's refusal direction to identify the attention heads most
responsible for safety behaviour, then ablates those heads. The idea of scoring
heads by their alignment with the refusal direction is the same idea as the
per-head term in LiberShift's attribution. RDSHA intervenes on activations at
inference; LiberShift attributes the mean write and then edits weights. The
per-head attribution is not new.

### *How Alignment Routes* (2026)

[arXiv:2604.04385](https://arxiv.org/abs/2604.04385)

This work localises a policy-routing circuit with direct logit attribution,
directional intervention, activation interchange and knockout. It reports that
the gate head contributes less than 1% of output direct logit attribution and
still controls the behaviour. That is the same dissociation LiberShift measures
between attributed load and causal effect. The circuit version is more detailed
and comes first.

### Multi-dimensional refusal, weighted directions, domain-specific ablation

[The Hidden Dimensions of LLM Alignment](https://arxiv.org/abs/2502.09674) ·
[WRDO](https://arxiv.org/abs/2610.29555) ·
[Not All Refusals Are Equal](https://arxiv.org/abs/2607.02714)

These works argue that refusal is not one direction: a dominant direction
coexists with smaller orthogonal directions, per-prompt directions can be
combined with learned weights, and refusal is distributed across layers in large
models. LiberShift's own measurement agrees: the cosine between `r[2]` and
`r[28]` is 0.04, and a single direction at the last plane leaves 58% refusal.
LiberShift still uses one mid-stack direction, applied everywhere. It is a
practical approximation, not a claim that refusal is one-dimensional.

## The comparison

### *Is Abliteration Getting Harder? Not Uniformly* (2026)

[arXiv:2611.17846](https://arxiv.org/abs/2611.17846)

This study compares direction-based methods (difference-in-means, MPOA, SOM,
J-space) and ARA on five release pairs under one prompt set, one nominal search
budget, one KL cap and one selection rule, with 972 held-out harmful prompts and
a WildGuard judge. It sets the current bar for an abliteration comparison.
LiberShift's comparison is smaller: two models, one baseline tool, keyword
refusal. The numbers here are a first result, not the last word.

## Artifact formats

### LoRA and task arithmetic

[LoRA](https://arxiv.org/abs/2106.09685) ·
[Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

A rank-r weight delta stored as factors is a LoRA adapter. A weight difference
between a base model and an edited model is a task vector. LiberShift's patch is
a rank-1 adapter restricted to `o_proj` and `down_proj`, plus the per-atom norms
needed to replay the edit exactly. The factorisation is standard; the replay
guarantee is the engineering.

## The combination

I could not find this exact combination published as one procedure:

1. a projected difference-of-means direction,
2. norm-preserving rank-1 orthogonalization at every layer,
3. the plane chosen by calibration on a disjoint split,
4. the edit exported as rank-1 factors with exact replay.

Every adjacent combination exists. Arditi supplies 1 and 2 without norm
preservation and with a larger search. grimjim supplies 2 with a heuristic layer
choice. Heretic searches the whole parameter space. RDSHA attributes to heads
without editing weights.

A combination of known parts is a configuration. It does not create a capability
that the parts do not already imply, and this repository makes no theoretical
claim. What makes the configuration worth reporting is the measured Pareto point
and the fixed cost, not the join itself.

The redirect is a different kind of claim. It generalises the removal update,
which is its `gamma = 0` case, so it inherits the removal literature's rank and
norm guarantees and adds a target direction from a measured gradient. What I did
not find published is the combination of a sparse attributed atom set, a
norm-preserving rank-1 redirect of each selected column, and a target taken from
the gradient of an acceptance score. The measured result is the gain: at 35,043
atoms the redirect reaches 0.11 refusal against 0.52 for removal, and at
`gamma = 0.75` it matches the dense edit's 0.060 refusal with 14% of its atoms,
at 3.7 times the KL. The better configuration keeps the removal dense and puts
the target on the top 5% of the efficiency ranking: 0.033 held-out refusal at
0.0447 KL against the dense edit's 0.097 at 0.0296. It also carries the only
category control in the study (0.00 cyber against 0.35 elsewhere).

The study does not isolate the contribution of each part. A controlled ablation
would compare, at the same held-out prompts: Arditi's edit without norm
restoration, grimjim's edit with a fixed plane, and this pipeline. The plane
sweep suggests that plane choice explains much of the gap between 0.06 and 0.24
refusal, so the calibration is doing real work, but the size of its contribution
next to norm preservation is not measured here. That is the most useful next
experiment.

## What would count as a theoretical contribution

Not done here, and worth stating so the boundary is clear:

- a bound on residual refusal after a rank-k partial ablation, as a function of
  the load left in the matrix's range;
- a predictor of the refusal threshold from the attribution, rather than a
  post-hoc curve;
- a proof that the per-layer direction rotation forces a dense edit, rather
  than the measurement that it does.

The empirical curve in `docs/results.md` is the start of the first item, not the
item itself.

## Confidence

| claim | status | confidence |
| --- | --- | --- |
| one-pass atom sparsity fails at 90% load (0.44 refusal) | new measurement | high for this model |
| a second round at 90% load reaches 0.21 | new measurement, one run | medium |
| the full matrix reaches 0.08 | confirmation of the default practice | high |
| the propagation score picks plane 22, optimum is 18 | new measurement | high for this model |
| attention carries 11% of the write and 31 of the 92 refusal points the edit removes | new quantification, known direction | medium |
| 30 prompts do not separate planes on 0.5B | new measurement | high for this model |
| the attribution identity closes to 4.8e-4 | new as a stated gate | high |
| 18.4 s to produce the model | new engineering, one GPU | high for this hardware |
| the patch replays to zero logit difference | new engineering | high |
| the pre-norm last plane matters under transformers 5.x | new engineering detail | high for 5.23, version-dependent |
| the removal edit is new | false | — |
| the redirect converts more than removal at the same atoms | new measurement | high for this model |
| the acceptance gradient is nearly orthogonal to the refusal direction (cosine at most 0.153) | new measurement | high |
| the gradient target converts more per unit KL than `-r` on the 100-prompt slice | new measurement | high for this model |
| the gradient target wins held out only above KL 0.10, and ties at 0.04 | new measurement, 300 prompts | medium |
| over half of the gradient target's KL is the intended redistribution | new measurement, post-hoc reading | medium |
| the redirect is category-selective (0.00 cyber against 0.35 elsewhere) | new measurement, one category, one budget | medium |
| a hybrid that removes the atoms with the worst conversion per harmless write cuts KL without a measurable refusal cost | new measurement, two models, one proxy | medium |
| dense removal with the target on the efficient 5% cuts held-out refusal from 0.097 to 0.033 at 1.51 times the KL | new measurement, two models, weaker on the smaller | high for the 1.5B model |
| the target's cost is set by the number of atoms that carry it, not the number removed | new measurement, one model | high |
| a target that moves the harmless distribution less converts less, in proportion | new measurement, four variants | medium; four variants, one operating point |
| a 128-prompt batch is 1.85 times faster on evaluation, with KL within 4e-4 | new engineering | high for this hardware |
| a phrase target can replace the acceptance direction | new measurement, one model | rejected: no lambda beats the acceptance gradient, and the phrase never appears |
| the `gamma > 0` redirect with a measured target is new | not found in the cited literature | medium; absence of evidence, one search |

## References

1. A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda.
   *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024.
   arXiv:2406.11717.
2. J. Lai (grimjim). *Projected Abliteration.* Hugging Face, 2025.
3. J. Lai (grimjim). *Norm-Preserving Biprojected Abliteration.* Hugging Face,
   6 November 2025.
4. P. E. Weidmann. *Heretic: Training-free language model modification.* GitHub,
   2025.
5. FailSpy. *abliterator.* GitHub.
6. C. Huang, Z. Zhang, J. Yue, Q. Li, C. Zhang, T. Liu. *Safety Alignment Should
   Be Made More Than Just A Few Attention Heads.* arXiv:2508.19697, 2025.
7. G. N. Frank. *How Alignment Routes: Localizing, Scaling, and Controlling
   Policy Circuits in Language Models.* arXiv:2604.04385, 2026.
8. *The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of
   Orthogonal Safety Directions.* arXiv:2502.09674, 2025.
9. G. Evseev et al. *WRDO: Weighted Refusal Direction Optimization in Aligned
   Language Models.* arXiv:2610.29555, 2026.
10. V. Hadetskyi, D. Pasquini, A. Sorokin. *Not All Refusals Are Equal.*
    arXiv:2607.02714, 2026.
11. R. J. Young. *Is Abliteration Getting Harder? Not Uniformly.*
    arXiv:2611.17846, 2026.
12. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.*
    arXiv:2106.09685, 2021.
13. G. Ilharco et al. *Editing Models with Task Arithmetic.* arXiv:2212.04089,
    2022.
14. T. McGrath et al. *The Hydra Effect: Emergent Self-repair in Language Model
    Computations.* arXiv:2307.15771, 2023.
