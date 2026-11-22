# Numbers behind the figures

## Qwen2.5-1.5B-Instruct

| model | report refusal | held-out refusal | KL | PPL |
| --- | --- | --- | --- | --- |
| base | 1.000 (70/70) | 0.870 (261/300) | 0.0000 | 11.54 |
| LiberShift | 0.071 (5/70) | 0.097 (29/300) | 0.0296 | 11.88 |
| LiberShift+target | 0.043 (3/70) | 0.033 (10/300) | 0.0447 | 11.88 |
| Heretic MPOA | 0.229 (16/70) | 0.210 (63/300) | 0.0285 | 11.57 |
| Heretic ARA | 0.000 (0/70) | 0.007 (2/300) | 0.6120 | 13.61 |

| model | arc_easy | piqa | winogrande |
| --- | --- | --- | --- |
| base | 0.733 | 0.757 | 0.643 |
| LiberShift | 0.730 | 0.757 | 0.640 |
| LiberShift+target | 0.720 | 0.757 | 0.650 |
| Heretic MPOA | 0.730 | 0.753 | 0.633 |
| Heretic ARA | 0.683 | 0.723 | 0.617 |

## Qwen2.5-0.5B-Instruct

| model | report refusal | held-out refusal | KL | PPL |
| --- | --- | --- | --- | --- |
| base | 0.900 (63/70) | 0.727 (218/300) | 0.0000 | 17.69 |
| LiberShift | 0.100 (7/70) | 0.167 (50/300) | 0.0862 | 18.07 |
| LiberShift+target | 0.057 (4/70) | 0.093 (28/300) | 0.1478 | 18.11 |
| Heretic MPOA | 0.086 (6/70) | 0.137 (41/300) | 0.0330 | 17.73 |
| Heretic ARA | 0.000 (0/70) | 0.000 (0/300) | 3.1484 | 106.61 |

| model | arc_easy | piqa | winogrande |
| --- | --- | --- | --- |
| base | 0.583 | 0.713 | 0.557 |
| LiberShift | 0.567 | 0.703 | 0.550 |
| LiberShift+target | 0.570 | 0.707 | 0.557 |
| Heretic MPOA | 0.567 | 0.710 | 0.570 |
| Heretic ARA | 0.397 | 0.603 | 0.530 |

## The redirect

Qwen2.5-1.5B, plane 18, 35,043 atoms. Refusal is the 100-prompt evaluation slice or the 300 held-out prompts.

| target | gamma | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- |
| -r | 0.3 | 0.380 | 0.283 | 0.0309 | 11.57 |
| -r | 0.5 | 0.330 | 0.243 | 0.0381 | 11.58 |
| -r | 0.75 | 0.250 | 0.213 | 0.0521 | 11.59 |
| -r | 1.0 | 0.150 | 0.183 | 0.0617 | 11.60 |
| gradient | 0.3 | 0.110 | 0.230 | 0.0406 | 11.57 |
| gradient | 0.5 | 0.080 | 0.197 | 0.0663 | 11.59 |
| gradient | 0.75 | 0.060 | 0.117 | 0.1103 | 11.62 |
| gradient | 1.0 | 0.070 | 0.130 | 0.1681 | 11.66 |
| gradient, orthogonalised | 0.3 | 0.110 | 0.257 | 0.0391 | 11.58 |
| gradient, orthogonalised | 0.5 | 0.110 | 0.180 | 0.0645 | 11.60 |
| gradient, orthogonalised | 0.75 | 0.060 | 0.127 | 0.1088 | 11.62 |
| gradient, orthogonalised | 1.0 | 0.060 | 0.117 | 0.1659 | 11.66 |

## Where the KL goes

| edit | KL | intended | elsewhere | intended share |
| --- | --- | --- | --- | --- |
| removal, 90% | 0.0197 | 0.0027 | 0.0170 | 13.5% |
| -r, gamma 0.5 | 0.0381 | 0.0047 | 0.0334 | 12.3% |
| -r, gamma 1.0 | 0.0617 | 0.0088 | 0.0529 | 14.3% |
| gradient, gamma 0.3 | 0.0406 | 0.0219 | 0.0187 | 53.9% |
| gradient, gamma 0.5 | 0.0663 | 0.0340 | 0.0323 | 51.3% |
| gradient, gamma 0.75 | 0.1103 | 0.0482 | 0.0620 | 43.7% |
| dense removal | 0.0296 | 0.0036 | 0.0261 | 12.1% |

## Category control

| atoms | strength | cyber refusal | other refusal | KL |
| --- | --- | --- | --- | --- |
| 5,000 | 2.0 | 0.955 | 0.987 | 0.004 |
| 20,000 | 2.5 | 0.682 | 0.769 | 0.033 |
| 50,000 | 2.5 | 0.000 | 0.346 | 0.201 |

## Hybrid (efficiency-ranked)

The bottom q of atoms by refusal load per unit of first-order harmless write is removed; the rest redirected.

| gamma | q | removal atoms | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0 | 0 | 0.110 | 0.230 | 0.0406 | 11.57 |
| 0.3 | 0.05 | 1752 | 0.100 | 0.227 | 0.0394 | 11.58 |
| 0.3 | 0.1 | 3504 | 0.110 | 0.227 | 0.0388 | 11.58 |
| 0.3 | 0.15 | 5256 | 0.070 | 0.213 | 0.0380 | 11.58 |
| 0.3 | 0.2 | 7009 | 0.090 | 0.223 | 0.0369 | 11.58 |
| 0.3 | 0.25 | 8761 | 0.090 | 0.210 | 0.0351 | 11.57 |
| 0.3 | 0.3 | 10513 | 0.100 | 0.243 | 0.0316 | 11.57 |
| 0.3 | 0.35 | 12265 | 0.080 | 0.237 | 0.0313 | 11.57 |
| 0.3 | 0.4 | 14017 | 0.100 | 0.217 | 0.0301 | 11.57 |
| 0.75 | 0 | 0 | 0.060 | 0.117 | 0.1103 | 11.62 |
| 0.75 | 0.05 | 1752 | 0.060 | 0.123 | 0.1076 | 11.62 |
| 0.75 | 0.1 | 3504 | 0.050 | 0.127 | 0.1023 | 11.62 |
| 0.75 | 0.15 | 5256 | 0.060 | 0.127 | 0.0945 | 11.63 |
| 0.75 | 0.2 | 7009 | 0.070 | 0.113 | 0.0902 | 11.61 |
| 0.75 | 0.25 | 8761 | 0.080 | 0.117 | 0.0790 | 11.60 |
| 0.75 | 0.3 | 10513 | 0.070 | 0.110 | 0.0733 | 11.60 |
| 0.75 | 0.35 | 12265 | 0.080 | 0.127 | 0.0651 | 11.60 |
| 0.75 | 0.4 | 14017 | 0.070 | 0.120 | 0.0605 | 11.60 |

## Continuous per-atom gamma

Every selected atom gets gamma * t(rank); the binary hybrid is the step.

| gamma | mapping | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- |
| 0.3 | linear | 0.170 | 0.250 | 0.0276 | 11.57 |
| 0.3 | square | 0.170 | 0.243 | 0.0248 | 11.57 |
| 0.3 | sqrt | 0.100 | 0.243 | 0.0317 | 11.58 |
| 0.3 | soft_q | 0.090 | 0.230 | 0.0327 | 11.57 |
| 0.75 | linear | 0.080 | 0.137 | 0.0473 | 11.59 |
| 0.75 | square | 0.070 | 0.160 | 0.0355 | 11.59 |
| 0.75 | sqrt | 0.060 | 0.127 | 0.0646 | 11.60 |
| 0.75 | soft_q | 0.060 | 0.140 | 0.0731 | 11.60 |

## The redirect on Qwen2.5-0.5B-Instruct

Plane 14, 16,894 atoms. The target is fitted on this model's own prompts.

| gamma | q | removal atoms | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0 | 0 | 0.120 | 0.257 | 0.1017 | 17.77 |
| 0.3 | 0.25 | 4224 | 0.110 | 0.253 | 0.0856 | 17.71 |
| 0.3 | 0.3 | 5068 | 0.120 | 0.253 | 0.0820 | 17.71 |
| 0.3 | 0.4 | 6758 | 0.100 | 0.237 | 0.0783 | 17.71 |
| 0.3 | 1 | 16894 | 0.480 | 0.457 | 0.0419 | 17.70 |
| 0.75 | 0 | 0 | 0.060 | 0.140 | 0.2830 | 18.01 |
| 0.75 | 0.25 | 4224 | 0.060 | 0.133 | 0.2127 | 17.75 |
| 0.75 | 0.3 | 5068 | 0.060 | 0.133 | 0.2084 | 17.75 |
| 0.75 | 0.4 | 6758 | 0.100 | 0.133 | 0.1780 | 17.72 |
| 0.75 | 1 | 16894 | 0.480 | 0.457 | 0.0419 | 17.70 |

## Dense removal plus a redirect of the efficient atoms

Qwen2.5-1.5B, plane 18, all 251,216 atoms. The atoms below the quantile q are removed; the rest carry the target at `gamma`. q=1 is dense removal.

| gamma | q | removed | redirected | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8 | 200,973 | 50,243 | 0.060 | 0.097 | 0.0300 | 11.87 |
| 0 | 0.9 | 226,094 | 25,122 | 0.060 | 0.093 | 0.0300 | 11.88 |
| 0 | 1 | 251,216 | 0 | 0.060 | 0.097 | 0.0296 | 11.88 |
| 0.3 | 0.45 | 113,047 | 138,169 | 0.030 | 0.030 | 0.1094 | 12.03 |
| 0.3 | 0.6 | 150,730 | 100,486 | 0.030 | 0.047 | 0.0981 | 11.98 |
| 0.3 | 0.7 | 175,851 | 75,365 | 0.020 | 0.053 | 0.0832 | 11.91 |
| 0.3 | 0.8 | 200,973 | 50,243 | 0.030 | 0.053 | 0.0633 | 11.89 |
| 0.3 | 0.85 | 213,534 | 37,682 | 0.030 | 0.053 | 0.0512 | 11.89 |
| 0.3 | 0.88 | 221,070 | 30,146 | 0.040 | 0.057 | 0.0454 | 11.89 |
| 0.3 | 0.9 | 226,094 | 25,122 | 0.030 | 0.047 | 0.0426 | 11.88 |
| 0.3 | 0.92 | 231,119 | 20,097 | 0.030 | 0.050 | 0.0400 | 11.88 |
| 0.3 | 0.95 | 238,655 | 12,561 | 0.030 | 0.047 | 0.0387 | 11.88 |
| 0.3 | 0.97 | 243,680 | 7,536 | 0.030 | 0.057 | 0.0358 | 11.87 |
| 0.3 | 0.99 | 248,704 | 2,512 | 0.030 | 0.053 | 0.0340 | 11.86 |
| 0.3 | 0.995 | 249,960 | 1,256 | 0.030 | 0.067 | 0.0329 | 11.86 |
| 0.45 | 0.9 | 226,094 | 25,122 | 0.020 | 0.040 | 0.0522 | 11.90 |
| 0.45 | 0.95 | 238,655 | 12,561 | 0.030 | 0.033 | 0.0447 | 11.88 |
| 0.45 | 0.97 | 243,680 | 7,536 | 0.030 | 0.047 | 0.0413 | 11.87 |
| 0.45 | 0.99 | 248,704 | 2,512 | 0.030 | 0.053 | 0.0371 | 11.86 |
| 0.6 | 0.95 | 238,655 | 12,561 | 0.030 | 0.037 | 0.0522 | 11.90 |
| 0.6 | 0.97 | 243,680 | 7,536 | 0.030 | 0.037 | 0.0464 | 11.87 |
| 0.6 | 0.99 | 248,704 | 2,512 | 0.030 | 0.057 | 0.0394 | 11.86 |
| 0.8 | 0.95 | 238,655 | 12,561 | 0.020 | 0.033 | 0.0657 | 11.93 |
| 0.8 | 0.99 | 248,704 | 2,512 | 0.030 | 0.043 | 0.0446 | 11.86 |

## The same family on Qwen2.5-0.5B-Instruct

Plane 14, 117,072 atoms.

| gamma | q | redirected | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9 | 11,707 | 0.080 | 0.160 | 0.0875 | 18.08 |
| 0 | 0.95 | 5,854 | 0.080 | 0.157 | 0.0853 | 18.08 |
| 0 | 1 | 0 | 0.080 | 0.167 | 0.0862 | 18.07 |
| 0.3 | 0.9 | 11,707 | 0.040 | 0.117 | 0.1521 | 18.11 |
| 0.3 | 0.95 | 5,854 | 0.030 | 0.130 | 0.1093 | 18.10 |
| 0.3 | 0.99 | 1,171 | 0.030 | 0.137 | 0.1003 | 18.09 |
| 0.45 | 0.9 | 11,707 | 0.050 | 0.103 | 0.1918 | 18.13 |
| 0.45 | 0.95 | 5,854 | 0.040 | 0.120 | 0.1271 | 18.10 |
| 0.45 | 0.99 | 1,171 | 0.040 | 0.137 | 0.1050 | 18.09 |
| 0.6 | 0.9 | 11,707 | 0.040 | 0.090 | 0.2432 | 18.18 |
| 0.6 | 0.95 | 5,854 | 0.040 | 0.093 | 0.1478 | 18.11 |

## Which atoms get the target

q=0.95, gamma=0.45. The same atom count is chosen three ways.

| ranking | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- |
| load per unit harmless write | 0.030 | 0.033 | 0.0447 | 11.88 |
| load | 0.020 | 0.030 | 0.0799 | 12.07 |
| harmless write | 0.030 | 0.033 | 0.1743 | 12.69 |

## Which direction the target points

q=0.95, gamma=0.45, all atoms. The target is the acceptance gradient, modified as named.

| target | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- |
| acceptance gradient | 0.030 | 0.033 | 0.0447 | 11.88 |
| orthogonal to r | 0.030 | 0.037 | 0.0431 | 11.89 |
| orthogonal to the mean harmless gradient | 0.020 | 0.030 | 0.0445 | 11.88 |
| orthogonal to the top 32 harmless directions | 0.040 | 0.080 | 0.0310 | 11.87 |
| per-layer gamma by measured conversion | 0.020 | 0.027 | 0.0501 | 11.88 |

## Batch size

Evaluation cost and result at four batch sizes, base model. The KL column is measured against a baseline taken at batch 32, so the nonzero values at larger batches are the size of the protocol difference, not a change in the model.

| batch | generate | first-token logits | measure | refusal, 100 | KL |
| --- | --- | --- | --- | --- | --- |
| 32 | 4.62 s | 0.34 s | 9.58 s | 1.000 | 0.0000 |
| 64 | 2.61 s | 0.32 s | 6.37 s | 0.990 | 0.0013 |
| 96 | 2.66 s | 0.35 s | 6.22 s | 0.990 | 0.0015 |
| 128 | 1.87 s | 0.33 s | 5.49 s | 0.990 | 0.0014 |

## Protocol parity

Every comparison checkpoint at both batch sizes, same harness.

| model | batch | refusal, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- | --- |
| base | 32 | 1.000 | 0.870 | 0.0000 | 11.54 |
| base | 128 | 0.990 | 0.860 | 0.0000 | 11.54 |
| dense removal | 32 | 0.060 | 0.097 | 0.0296 | 11.88 |
| dense removal | 128 | 0.060 | 0.100 | 0.0297 | 11.88 |
| Heretic MPOA | 32 | 0.270 | 0.210 | 0.0285 | 11.57 |
| Heretic MPOA | 128 | 0.270 | 0.200 | 0.0283 | 11.57 |
| Heretic ARA | 32 | 0.000 | 0.007 | 0.6120 | 13.61 |
| Heretic ARA | 128 | 0.000 | 0.007 | 0.6124 | 13.61 |

## Phrase weight

lambda = 0 is the acceptance gradient; raising lambda raises the phrase but never surfaces it.

| gamma | lambda | refusal, 100 | held out, 300 | KL | phrase logprob | starts | contains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0 | 0.100 | 0.230 | 0.0416 | -19.93 | 0/100 | 0/100 |
| 0.3 | 0.25 | 0.130 | 0.230 | 0.0390 | -18.73 | 0/100 | 0/100 |
| 0.3 | 0.5 | 0.140 | 0.237 | 0.0379 | -17.72 | 0/100 | 0/100 |
| 0.3 | 1 | 0.200 | 0.247 | 0.0336 | -16.35 | 0/100 | 0/100 |
| 0.3 | 2 | 0.310 | 0.307 | 0.0308 | -15.10 | 0/100 | 0/100 |
| 0.3 | 4 | 0.380 | 0.367 | 0.0296 | -14.46 | 0/100 | 0/100 |
| 0.75 | 0 | 0.070 | 0.123 | 0.1085 | -20.04 | 0/100 | 0/100 |
| 0.75 | 0.25 | 0.060 | 0.137 | 0.1087 | -18.37 | 0/100 | 0/100 |
| 0.75 | 0.5 | 0.060 | 0.140 | 0.0989 | -17.06 | 0/100 | 0/100 |
| 0.75 | 1 | 0.050 | 0.173 | 0.0882 | -14.83 | 0/100 | 0/100 |
| 0.75 | 2 | 0.200 | 0.203 | 0.0790 | -12.70 | 0/100 | 0/100 |
| 0.75 | 4 | 0.320 | 0.267 | 0.0737 | -11.45 | 0/100 | 0/100 |

## Time

| stage | seconds |
| --- | --- |
| load model | 0.74 |
| capture 800 prompts | 3.10 |
| attribute 28 layers | 1.18 |
| calibrate 6 planes x 30 prompts | 12.16 |
| apply edit | 0.22 |
| export checkpoint + patch | 1.02 |
| final evaluation (370 prompts) | 15.24 |
| Heretic MPOA, 40 trials, whole process | 141.3 |
| Heretic ARA, 40 trials, whole process | 398.4 |

## Qwen3.5-4B

32 layers, hidden 2560, attention atoms 'columns' (4096 per layer, layer types {'linear_attention': 24, 'full_attention': 8}), plane 20 of 32.

| model | eval, 100 | held out, 300 | KL | PPL |
| --- | --- | --- | --- | --- |
| base | 93/100 | 180/300 | 0 | 10.72 |
| dense removal | 61/100 | 116/300 | 0.0567 | 10.87 |
| target on the top 5%, gamma=0.75 | 49/100 | 69/300 | 0.1944 | 10.87 |
| Heretic MPOA, 40 trials | 55/100 | 88/300 | 0.1027 | 10.74 |
| Heretic ARA, 40 trials | 3/100 | 3/300 | 1.0832 | 13.57 |

Exported checkpoint after reload: 49/100, 69/300, KL 0.1944, PPL 10.87.
Heretic PR #211 reports for this model: MPOA still above 50/100 after abliteration; ARA 4/100 at 0.1396 KL (reported, not re-measured).
