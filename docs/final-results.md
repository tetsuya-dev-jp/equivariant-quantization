# Independent evaluation

Paired calibration improves some correctness and composition outcomes while
reducing agreement with the full-precision teacher. It should not be described
as uniform behavior preservation. The comparison is against augmented KD with
the same input rows, order, trainable scales and 96 optimizer updates.

## Evaluation boundary

Each condition evaluates 768 final bases and 384 larger-structure bases:
1,152 distinct problems, rendered into 6,144 inputs. Variants of a problem
stay together. The released report separates both splits and all three task
families; pooled results below weight every base equally.

Entity renaming and fact order are absent from gradient updates but present
in development selection. Compositions are absent from both fitting and
selection. Thus the composition endpoint is a separate secondary transfer
test. It is not evidence for primitive transformation families unseen to the
complete selection procedure.

Consistency C compares canonical option identities. Joint correctness J
requires both the original and transformed input to be correct. Consistently
wrong agreement W requires agreement on a wrong canonical option; C = J + W.
These rates first average variants within a base and then average bases.
Teacher fidelity compares the fitted model and BF16 on the same input.
Fit seeds are reported separately, not treated as extra evaluation problems.

## Qwen2.5-1.5B

On the pooled final and structural bases, the paired loss raises joint
correctness but lowers total consistency on held-out single transformations.
The seed-43 joint-correctness interval spans zero. Wrong agreement falls at
every seed. Table 1 uses paired minus same-seed augmented KD; all values are
percentage points. Intervals use 1,000 paired whole-base bootstrap resamples
at seed 4108 and are unadjusted, conditional on each fitted model.

Table 1. Qwen contrasts on held-out single transformations, 1,152 bases

| Fit seed | Delta C | Delta J [95% interval] | Delta W | Delta teacher fidelity |
|--:|--:|--:|--:|--:|
| 17 | -1.87 | +11.50 [8.55, 14.37] | -13.37 | -6.42 |
| 29 | -1.87 | +3.47 [0.82, 6.08] | -5.34 | -13.11 |
| 43 | -4.51 | +2.04 [-0.52, 4.77] | -6.55 | -9.29 |

For fully withheld compositions, consistency rises by 14.71–20.62 points and
joint correctness by 10.63–16.19 points across seeds; all six intervals lie
above zero. Wrong composition agreement rises by 4.08 and 9.42 points at
seeds 29 and 43, while seed 17 changes it by -0.17 points with an interval
spanning zero. Composition teacher fidelity falls by 3.39–12.63 points.
This is a mixed correctness/invariance tradeoff, rather than preservation of
the teacher's predictions.

The four-logit mean-drift correction does not account for the paired result.
On the final split, held-out joint correctness is 39.65% for W4 and 39.19%
after that correction, versus descriptive three-fit means of 47.96% for
augmented KD and 54.41% for paired calibration. The corresponding random-pair
and kernel-control means are both 47.48%. These are descriptive control
comparisons; their magnitudes are not paired-bootstrap intervals.

## Executed settings

Table 2. Common settings and evidence

| Item | Executed value | Evidence |
|:--|:--|:--|
| Models | Qwen2.5-1.5B-Instruct; SmolLM2-1.7B-Instruct; pinned revisions | Configs and run manifests |
| Quantization | Signed RTN W8/W4/W3, group 128; calibration W4; BF16 forward | Run manifests |
| Frozen parameters | Integer codes and pretrained weights, biases, head, embeddings, norms | Calibration implementation and fit manifests |
| Trainable deltas | Qwen 10,235,904; Smol 12,582,912 | Fit manifests |
| Training | AdamW lr 0.001; betas 0.9/0.999; weight decay 0; clip 1 | Fit manifests |
| Exposure | Four bases/eight prompts per update; 96 updates; 384 bases/768 prompts | Input and batch-order records |
| Fit seeds | 17, 29, 43 | Fit manifests |
| Strength search | 0.1, 1, 10; seed-17 development selection | Frozen selection JSON |
| Selected strengths | Qwen paired/random/kernel: 10/1/0.1; Smol: 10/0.1/10 | Frozen selections |
| Scoring | Official chat template, next-token A–D, batch 8, maximum 1,024 tokens without truncation | Configs/scorer |
| Runtime | Python 3.12, Torch 2.12.0+rocm7.14.0, Transformers 4.57.1, RX 9070 XT | environment.json |

See the [protocol](protocol.md), [calibration plan](calibration-plan.md), and
[reproduction guide](reproduce.md) for exact definitions, parameters and
commands. The [novelty audit](related-work.md) distinguishes this controlled
application from established consistency training and relational quantization.
