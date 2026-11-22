# LiberShift: method

How the edit is derived, and the measurements that constrain it.

LiberShift edits a transformer by changing the matrices that write into the
residual stream. Two edits are documented here. The removal edit deletes a
behavioural direction from every matrix that writes it; it is Arditi et al.'s
weight orthogonalization with norm preservation after grimjim. The redirect edit
keeps the atom and replaces the direction it writes with a measured target. The
trace supplies an exactness check and the measurements that rule out cheaper
edits. This document states both methods and the evidence; `related-work.md`
states which part came from where.

## 0. What is new, and what is reused

This repository holds two edits. Their status differs.

The removal edit in §5.1 is not new. Arditi et al. derive it in §4.1 of
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717), Eq. 5, for every matrix
that writes to the residual stream. grimjim adds the norm restoration in
[norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration),
rescaling each matrix row, and states that the update stays rank one. The layer
selection in §7 is Arditi's search, reduced from layer by position to six planes.

The redirect in §5.2 is new here. It keeps the component that removal deletes
and mirrors it into a chosen target direction:

```
d_f <- d_f - alpha_f r + gamma * alpha_f a_l
```

The atoms are the same sparse attributed set. The gate row, the up row and the
column norm stay fixed. `a_l` comes from measurement in two forms: the sign flip
`a = -r`, and the gradient of an acceptance score with respect to the residual
plane (§5.3). I have not found this rule published. Its nearest relatives are
the removal edit it generalises, inference-time activation addition, and concept
erasure; `related-work.md` states the boundary work by work. The claim here is
the measurement. A reader can write the update in one line, and what this
document reports is what the update does to refusal, KL and the held-out set.

The gain is specific. At 35,043 atoms, 14% of a dense abliteration, the redirect
lowers refusal from 0.52 to 0.11 on the 100-prompt evaluation slice when the
target is the acceptance gradient, and to 0.15 with the sign flip. It is the
only edit here with a category control, and the KL split shows that over half of
its damage at the efficient operating point is the accept/refuse redistribution
the edit is asked to make. Those results are new measurements of a new rule, not
a re-description of abliteration.

What this document adds:

1. a measured atom-sparsity curve, including a second attribution round, that
   shows why a subset of atoms does not replace the full matrix under removal
   (§4);
2. a measured refutation of the trace-only plane score (§7);
3. the two-path load against causal effect, quantified (§4);
4. a calibration split whose failure on the 0.5B model is reported (§7);
5. an exact attribution identity used as a gate (§3);
6. the redirect, its two targets, and the measured tradeoff against removal at
   the same atoms (§5.2, §5.3);
7. a split of the harmless-prompt KL into the accept/refuse redistribution the
   method intends and the disturbance elsewhere, which changes how the
   redirect's damage reads (§5.3);
8. a hybrid that removes the atoms with the worst conversion per unit of
   harmless write, which cuts KL without a measurable refusal cost (§5.4).

The removal combination is a configuration of known parts. The redirect is a new
rule on the same part. This document makes no theoretical claim about either.

## 1. The object being edited

A decoder block writes into the residual stream through a sum of parts. With
`h[l]` the residual plane entering block `l`,

```
h[l+1] = h[l] + a[l] + f[l]
```

where `a[l]` is the attention write and `f[l]` is everything after it. The
contract is additive and holds exactly: measured relative error 3e-3 in
bfloat16, which is the precision of the forward pass.

Both writes decompose into atoms, and both decompositions are exact:

```
a[l] = sum over heads h of  W_O[:, h] @ o_h
f[l] = sum over features f of  c_f * d_f
```

`o_h` is a block of the input to `o_proj` and `c_f` is a row of the input to
`down_proj`. These are values the model already computes. Measured relative
error: 2e-3 for attention, 4e-3 for the FFN.

The planes were cross-checked against an independent f32 engine: its layer dump
and HF transformers agree to better than 3e-6 relative on every pre-norm plane
(`../results/parity.json`; the optional script needs that engine built). The
last plane needs care. Transformers 5.x returns it after the model's closing
norm, so it is not in the same space as the planes below it. The capture
reconstructs it as `h[L-1] + a[L-1] + f[L-1]`, which matches the independent
dump to 2.2e-6.

Two consequences drive everything below:

1. A direction in the residual stream is a sum of contributions, and each
   contribution has a name.
2. The FFN write is linear in the matrix `D` whose columns are `d_f`. Removing
   a direction from `D` removes it from the range of the map `c -> D c` for
   every input, not just for the prompts that were measured.

### Hybrid decoders

Qwen3.5-4B mixes two attention families in one stack: 24 of 32 layers are
linear attention, whose output projection is `linear_attn.out_proj`, and 8 are
full attention with `self_attn.o_proj`. The families do not share one head
layout across the stack, so no single head decomposition covers every layer.
When the capture sees layers that disagree this way, the attention atom is
demoted: one input channel of the attention output projection is one atom,
`head_dim = 1`, and the hook attaches to whichever output projection the layer
names. The FFN side is unchanged. Both decompositions hold per layer regardless
of family; the additive contract on Qwen3.5-4B shows the same bfloat16 noise as
the dense models (4e-3 worst plane). The capture batch for such a model is 12
instead of 32: the chunked gated-delta-rule transients of the linear-attention
kernel exceed a 16 GB GPU at the protocol batch.

## 2. The refusal direction

For each plane, take the last prompt position of 400 harmful and 400 harmless
prompts:

```
r[l] = normalize( mean_bad[h[l]] - mean_good[h[l]] )
```

then subtract the component along `normalize(mean_good[h[l]])`, as in projected
abliteration. The separation is a standardised mean gap (Cohen's d): 2.8 at
plane 1 rising to 8.2 at plane 28 on Qwen2.5-1.5B-Instruct.

## 3. Attribution

The change in mean projection onto `r` caused by atom `u` is

```
lambda_u = (mean_bad[c_u] - mean_good[c_u]) * (w_u . r)
```

with `c_u` the atom's activity and `w_u` its write vector. The lambdas of a
layer sum to that layer's measured change in projection, because projection is
linear and the write is a sum. Measured error: 4.8e-4 relative, every layer, both
paths. This identity is the method's correctness gate; if it failed, no number
in the study would mean anything.

Measured attribution, Qwen2.5-1.5B-Instruct:

| quantity | value |
| --- | --- |
| refusal write carried by the FFN | 348 of 391 positive load (89%) |
| refusal write carried by attention | 43 of 391 positive load (11%) |
| highest-load layer | 22 (largest per-layer value, 22.3) |
| share of FFN load in the top 1% of features | 40% |
| share of FFN load in the top 10% of features | 73% |

The 391 total is the sum of positive load. Negative contributions cancel part of
it, so the signed per-layer sum is 208.9. The 89/11 split describes the positive
load.

`cost` of an atom is the squared Frobenius norm of the change its ablation
makes, so costs add across layers and compare between heads and features.

## 4. Three measured limits on what attribution can claim

**Atom sparsity does not replace the full matrix.** Removing `r` from a subset
of atoms leaves `r` in the range of `D`, because thousands of other columns
remain. Any input that activates them can produce `r` again. The measurement
agrees: an edit covering 90% of the attributed load (34k of 251k atoms) leaves
refusal at 0.44; a second attribution round at the same target reaches 0.21;
covering every atom leaves 0.08. Re-selection helps, and still does not close
the gap. Refusal behaves like a threshold, so a partial reduction of a mean does
not remove it.

**The redirect changes that conclusion for conversion, and only partly.** The
same 90% selection, redirected at `gamma = 0.75` toward the acceptance gradient,
reaches 0.06 refusal on the 100-prompt evaluation slice and 0.117 held out,
against 0.52 and 0.44 for removal at the same atoms. The dense edit reaches 0.06
and 0.10 with 7 times the atoms and 3.7 times less KL. Dropping an atom's write
leaves the direction available to the columns that remain; re-pointing the write
does not, because the redirected atoms now write against it. Sparsity buys
conversion once the edit replaces the target. It does not buy the dense edit's
KL.

**The direction rotates with depth.** The cosine between `r[2]` and `r[28]` is
0.035; neighbouring planes are close (0.914 between `r[27]` and `r[28]`); the
leading singular direction of the 28 unit directions carries 32% of their summed
squared projection, and ten carry 85% (uncentered SVD; centering gives 27% and
81%). There is no single refusal direction in this model
(`../results/analysis/direction_geometry.json`).

**Causal effect does not follow the mean write.** Attention carries 11% of the
attributed load and the FFN carries 89%. The full edit removes 92 points of
refusal on this split: 61 from the FFN edit and 31 from the attention edit.
Attention's share of the effect is three times its share of the write. A layer
can write the direction and not matter; a layer can matter beyond its load.

## 5. The edit unit

### 5.1 Removal

The three limits above leave one honest unit for removal: the whole matrix, rank
one.

```
M <- M - s * r (r^T M)
```

followed by rescaling each atom's write vector to its original norm. `s` is 1.0
by default, which removes the direction exactly. The change is rank one per
matrix, so the edit is a rank-1 update. The norm restoration is exact here
because the method writes final weights rather than an adapter.

This is Arditi et al.'s weight orthogonalization with the norm-preservation
idea from grimjim, applied per atom. The edit is prior art. What the
measurements above add is why the tested alternatives do not replace the whole
matrix: a atom subset leaves `r` in the range, and a layer subset leaves other
layers writing it.

This edit has the property that matters: after it, the matrix cannot produce
`r` for any input.

### 5.2 Redirect

Removal asks what an atom writes and deletes the answer. The redirect asks the
same question and replaces the answer. For one atom of layer `l`:

```
alpha_f = d_f . r
d_f'    = d_f - alpha_f r + gamma * alpha_f a_l
```

`d_f` is column `f` of `down_proj`, or the matching block of `o_proj` for a
head. The atom keeps its gate row, its up row and its norm, so it fires on the
same inputs with the same strength. Only its edge target moves.

`gamma` scales the replacement against the component it replaces. `gamma = 0`
recovers removal exactly, and `gamma = 1` gives the new component the same
magnitude as the old one. `gamma = 1` with `a = -r` is the sign flip,
`d_f - 2 alpha_f r`, which reverses the atom's refusal write and restores the
norm exactly. The pilot and `docs/results.md` call that strength 2.0, because
they scale the removed component rather than the added one.

Two properties follow from the algebra:

1. the change is rank one per column, so the patch stays one direction per
   layer plus one coefficient per atom;
2. the sum of the added acceptance components over a layer is `gamma * a_l`
   times that layer's signed refusal load, so the push inherits the measured
   load without a second calibration.

The stub this does not solve is attention. `o_proj` is not one of the matrices
the patch covers, so the head half of a selection is edited in memory only and
does not ship. The measured head-only redirect moves little: 10 heads at
`gamma = 1` leave 0.85 refusal, and at `gamma = 2` 0.71. That is a sparse head
set, not the full attention edit, which the removal study measures at 31 of the
92 refusal points. The FFN-first order follows from the effect, not from the
tooling.

### 5.3 The acceptance target

`a = -r` points away from refusal. It does not point toward acceptance, and the
measurement separates the two: at `gamma = 1` the sign flip reaches 0.15 refusal
on the 100-prompt evaluation slice and 0.18 held out, where removal at the same
35,043 atoms stays at 0.52 and 0.44.

The acceptance direction is defined and measured. For one prompt the acceptance
score is

```
s = logsumexp(logit[accept set]) - logsumexp(logit[refuse set])
```

at the last prompt position. The token sets are read off the model's own top-8
next tokens. Harmful prompts put "I", "As", "Sorry" and "Unfortunately" first on
every prompt; harmless prompts put "The", "Here" and "Sure" first. The sets
define the score and take no part in atom selection.

`ds/dh[p]` is the direction in which plane `p` must move to raise acceptance,
including the effect of every later block. Block `l` writes into plane `l + 1`,
so `a_l` is the unit gradient at plane `l + 1`.

The gradient is nearly orthogonal to the refusal direction: cosine 0.028 at
plane 1, 0.153 at the maximum inside the stack (plane 13), and below 0.05
through plane 8. The two targets are therefore different edits and not one edit
at two scales. Removing the component along `r` from the gradient changes
nothing measurable, for the same reason.

`gamma` is the redirect's one free parameter, and 0.3 is the efficient point
measured here. The gradient target raises the score 37% further than `-r` at the
same `gamma` (10.42 against 7.60 on the fit prompts) and lowers refusal from
0.38 to 0.11 at 1.3 times the KL.

The gate and up matrices stay untouched by both edits. They select which
features fire, and changing them changes the model's activation pattern rather
than the direction it writes.

A phrase target does not beat the acceptance gradient. With
`a = unit(lambda * phrase + base)` and `base` the acceptance gradient,
`lambda = 0` gives 0.100/0.230 at `gamma = 0.3` and 0.070/0.123 at
`gamma = 0.75`; `lambda = 4` gives 0.380/0.367 and 0.320/0.267. The phrase
logprob rises from -19.9 to -11.5, but the literal phrase appears in 0 of 100
responses at every setting. The model writes "Sure, here's..." instead. Language
follows the distribution, not the requested string; the phrase target is
rejected.

### 5.4 Which atoms carry the target

The redirect's extra KL is not spread evenly. Rank every atom by its refusal
load per unit of first-order harmless write: for a feature,
`load / (|alpha| * rms(c))`; for a head, the analogue with `o_h . beta_h`.
Remove the refusal component from the whole edit, and give the target to the
top fraction only.

Two quantities are separate, and the measurements keep them apart. **Removal**
is what kills refusal: orthogonalising a write against `r` is a property of the
map, and the atom count that matters is the number removed. **The target** is
what converts refusal into acceptance, and its cost is set only by the number of
atoms that carry it. The shipped edit removes all 251,216 atoms and puts the
target on the top 5%, which is 12,561 atoms.

| target on | atoms | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- |
| none, dense removal | 0 | 0.060 | 0.097 | 0.0296 | 11.88 |
| the top 0.5%, `gamma = 0.3` | 1,256 | 0.030 | 0.067 | 0.0329 | 11.86 |
| the top 5%, `gamma = 0.45` | 12,561 | 0.030 | 0.033 | 0.0447 | 11.88 |
| the top 20%, `gamma = 0.3` | 50,243 | 0.030 | 0.053 | 0.0633 | 11.89 |
| the top 55%, `gamma = 0.3` | 138,169 | 0.030 | 0.030 | 0.1094 | 12.03 |

The top 5% takes held-out refusal from 0.097 to 0.033 at 1.51 times the KL and
the same perplexity. The top 0.5% takes it to 0.067 at 1.11 times the KL, which
is the cheapest useful edit in the study. Past 5% the curve flattens: at fixed
`gamma = 0.3`, moving the target from 5% to 55% of the atoms takes held-out
refusal from 0.047 to 0.030 and KL from 0.0387 to 0.1094.

The ranking is measured against its alternatives at the same atom count. The
same 12,561 atoms cost 0.0447 KL chosen by efficiency. Choosing them by load
alone costs 1.8 times that KL, and choosing them by the largest harmless write
costs 3.9 times, at the same refusal. The ratio is the signal.

The target direction is also measured against its alternatives, and they do not
help. Orthogonalising it against `r` changes nothing because the two directions
are already nearly orthogonal (cosine at most 0.153). Orthogonalising it against
the mean harmless top-1 gradient changes KL by 0.4%. Projecting out the top 32
directions of the harmless logit Jacobian cuts KL by 31% and gives back 2.4
times the refusal. Allocating `gamma` per layer by measured acceptance gained
per unit KL moves two prompts for 12% more KL. Conversion and harmless movement
share the same axes, so the frontier is close to a ray.

The fixed-set hybrid, which was the first form of this rule, is a special
case: it took 35,043 atoms selected by load per unit cost and applied the same
rule inside that set.

At `gamma = 0.3`, `q = 0.25` cuts KL from 0.0406 to 0.0351 (13.7%), with refusal
0.090 against 0.110 on the 100-prompt slice and 0.210 against 0.230 held out.
At `gamma = 0.75`, `q = 0.30` cuts KL from 0.1103 to 0.0733 (33.5%), with 0.110
held out against 0.117 and 0.070 on 100 against 0.060. The refusal differences
are inside sampling error; the KL cut is the measured gain.

The policy replicates on 0.5B at plane 14 with 16,894 atoms: `q = 0.40` at
`gamma = 0.3` gives 0.100/0.237/0.0783 against 0.120/0.257/0.1017, and `q = 0.30`
at `gamma = 0.75` gives 0.060/0.133/0.2084 against 0.060/0.140/0.2830. The
target is fitted on the 0.5B model's own prompts; this is a replication, not a
transfer of the 1.5B target.

Ranking by the largest harmless write instead is worse at both operating
points: at `gamma = 0.75` inside the fixed set, removing the top 10% by raw
damage raises held-out refusal to 0.203 against 0.117, because those atoms also
carry refusal load. The ratio is the signal.

The binary hybrid is a special case of a per-atom `gamma`: removal is
`gamma_f = 0`, redirect is `gamma_f = gamma`. A `.vlp` can represent it as two
disjoint redirect operations per layer, one with `gamma = 0`; `taa-patch/3`
would need a per-atom gamma array, or two disjoint entries, to ship it.

A continuous `gamma_f` was tested and does not improve the frontier. At
`gamma = 0.3` the best continuous point is 0.090/0.230/0.0327 against the binary
0.090/0.210/0.0351. At `gamma = 0.75` the best continuous point matches the
binary at 0.127 held out and KL 0.065, while the binary's q=0.30 point keeps
0.110 against the continuous soft point's 0.140 at the same KL. The step removes
the low-efficiency target component entirely, while a small gamma on every atom
pays KL everywhere. A per-atom KL gradient cannot improve the ranking at first
order because the KL is at a minimum at the base model; a second-order (Fisher
or Gauss-Newton) attribution is the next signal to try.

## 6. Which layers

Every layer that writes the direction must be edited, because each layer's
write adds its own component and the surviving layers still cross the refusal
threshold. grimjim argues the same from the [Hydra effect](https://arxiv.org/abs/2307.15771):
an ablated layer is partly restored by the others. The measurement here is
direct: with the plane-22 direction, editing 1 layer leaves refusal at 0.96, 7
layers 0.92, 10 layers 0.69, 14 layers 0.39, all 28 layers 0.24. The curve
decreases monotonically and no layer is decisive on its own.

So LiberShift edits every layer and gains no sparsity from layer selection. The
sparsity it does have is structural: the change per matrix is rank one, and the
ignored matrices (q, k, v, gate, up, embeddings) are bit-identical.

## 7. Which plane

The ablation direction is one plane's `r[p]`, applied at every layer. The
choice of `p` is the method's only free parameter. Arditi et al. choose the
source layer by searching layer by position, filtering candidates by KL
divergence and refusal score, and pruning the last 20% of layers. This reduces
the same idea to six planes and one metric.

A trace-only rule was tried and failed. The propagation score

```
S(p) = sum over l > p of  load[l] * max(0, cos(r[p], r[l+1]))
```

ranks plane 22 first and plane 18 fourth. The measured optimum in the sweep is
plane 18 (0.06 refusal) and the score's own top pick, plane 22, gives 0.24.
Every plane from 17 to 21 is usable, so the score finds the region and cannot
separate the region's members.

LiberShift therefore calibrates `p` on 30 harmful prompts that are disjoint
from the report prompts, taking the plane whose keyword refusal rate is lowest.
The calibration takes 12.2 seconds for six candidates. On the 1.5B model it
reproduces the full-evaluation optimum. On the 0.5B model it does not: planes
14, 16 and 18 tie at one refusal in thirty, and the held-out rates for the same
edit span 0.11 to 0.17 across planes 14 to 20. Thirty prompts do not separate
candidate planes on the smaller model. This is a search, and it is stated as
one: six candidates, one parameter, 30 prompts. Heretic searches nineteen
parameters with 40 full evaluations.

## 8. Verification

Three checks close the loop:

1. The attribution identity, before any edit (4.8e-4).
2. The projection readout after the edit: the harmful set's mean projection
   onto the final plane's direction, next to the harmless set's. A single
   plane-18 edit leaves 11.5 against -0.6 at plane 28, which is not zero, and
   refusal still falls to 7% on the report split and 10% on the held-out set.
   The projection is a diagnostic, not the target.
3. Held-out prompts: 300 harmful prompts from two other datasets that no method
   saw, plus a capability suite. The removal numbers come from the 70 report
   prompts and these 300, never from the calibration prompts; the redirect uses
   the full 100-prompt evaluation slice because it has no calibration split.
4. The acceptance score before and after a redirect. The mean score on the fit
   prompts moves from -8.56 to +1.86 at `gamma = 0.3`. A redirect that scored
   worse than removal would be a different edit from the one described here.
5. The exported model. Reload it from safetensors and measure again. The
   redirect at `gamma = 0.3` gives 0.110 refusal, 0.230 held out, 0.0406 KL and
   11.57 perplexity, the same numbers as the in-memory edit, and the operator
   patch replays to 0.0 logit difference.

## 9. Procedure

1. Capture per-layer residual planes, attention writes, FFN writes, per-head
   context vectors and per-feature activations for 400 harmful and 400 harmless
   prompts.
2. Compute `r[l]` for every plane, projected against the harmless mean.
3. Attribute refusal load to every head and feature; verify the identity.
4. Calibrate the edit plane on 30 harmful prompts disjoint from the report set.
5. Apply `M <- M - r (r^T M)` to `o_proj` and `down_proj` at every layer, with
   exact atom-norm restoration.
6. For the redirect: differentiate the acceptance score on the same 400 harmful
   prompts and take `a_l` as the unit gradient at plane `l + 1`. Apply
   `d_f <- d_f - alpha_f r + gamma alpha_f a_l` to the selected atoms, with the
   same atom-norm restoration. The gradient costs 2.5 s for 400 prompts
   (`results/redirect/gradient_timing.json`).
7. Export the weights and a patch. The removal patch stores one direction per
   layer plus one norm scale per atom. The redirect patch stores one direction,
   one target, and one index and norm scale per atom, and replay recomputes
   `d_f - alpha_f r + gamma alpha_f a` from the base column. Re-apply either
   patch to a pristine checkpoint and confirm zero logit difference.
8. Report refusal on disjoint and held-out prompt sets, KL, perplexity and
   capability.

Cost on one RTX 5080 Laptop for a 1.5B model, measured with a synchronised
clock on an idle GPU: 3.1 s to capture 800 prompts, 1.2 s to attribute 28
layers, 12.2 s to calibrate six planes, 0.2 s to edit, 1.0 s to export, 0.7 s
to load. That is 18.4 s to produce the model, against 141 s for a 40-trial
Heretic MPOA run and 399 s for a 40-trial ARA run on the same GPU. The redirect
replaces the 12.2 s calibration with a 2.5 s gradient and needs no plane search,
so its whole pipeline is 8.7 s.

## 10. Predictions and outcomes

| prediction | test | outcome |
| --- | --- | --- |
| the write decomposes exactly into atoms | attributed load sum vs measured projection | holds, 4.8e-4 |
| refusal write concentrates in few features | load concentration | partly, top 1% holds 40% |
| sparse atoms remove refusal cheaply | 90%-load sparse edit, one pass and two rounds, against the dense edit | fails, 0.44 and 0.21 against 0.08 |
| one direction at the last plane suffices | ablation with `r[28]` everywhere | fails, 0.58 refusal at zero projection |
| a trace score picks the best plane | propagation score vs sweep | fails, picks 22 where 18 is best |
| over-ablation beats exact removal | strength 1.0 to 1.45 | fails, costs 9x KL for 4 prompts out of 100 |
| the edit fits in kilobytes | rank-1 patch and replay | holds, 754 KB against 3.1 GB, replay exact |
| FFN carries most of the write | path ablation | holds, 89% of load |
| the acceptance direction is the negative refusal direction | cosine of the acceptance gradient against `-r` | fails, 0.028 at plane 1, 0.153 at the maximum inside the stack |
| replacing the removed component converts more than deleting it | redirect against removal at the same 35,043 atoms | holds, 0.15 against 0.52 on the 100-prompt slice, 0.18 against 0.44 held out |
| a measured target converts more per unit KL than a mirrored one | `-r` against the acceptance gradient at `gamma = 0.3` | holds, 0.11 against 0.38 refusal at 1.3 times the KL |
| a sparse redirect matches the dense removal edit | 35k redirected against 251k removed | fails held out, 0.23 at `gamma = 0.3` and 0.12 at `gamma = 0.75`, against 0.10 |
| the redirect is selective by category | cyber-only redirect at 50k atoms | holds, 0.00 cyber against 0.35 other, at KL 0.20 |
| a phrase target replaces the acceptance direction | phrase-only, and phrase with the refusal contrast | partly: phrase-only does not convert; phrase+contrast matches held-out and KL at `gamma = 0.75` |
| a hybrid that removes the worst redirects improves the Pareto | efficiency-ranked removal of q=0.25 at `gamma = 0.3` and q=0.3 at `gamma = 0.75` | partly, KL 13.7%/33.5% lower; refusal inside sampling error; dense still wins held-out per KL |
| the redirect fits in kilobytes and replays | operator patch, replay, reload | holds, 0.5 MB for 35,043 atoms, 0.0 logit difference, reloaded metrics identical |

One prediction is absent because its answer is known in advance: the edit is not
new. It is Arditi et al.'s Eq. 5 with norm preservation after grimjim. The table
lists what the measurements decided.

The failures are the substance of the result. They say that the mean-based
attribution describes what the model writes, and that behaviour needs the
direction removed structurally, at every layer, at a plane chosen by
measurement rather than by any of the summary statistics tried here.
