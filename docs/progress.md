# Experiment record

8 November 2026. Authoritative measurements are the saved configurations,
per-input predictions, fit manifests and comparison JSON. This record links
the prospective decisions to execution; the manuscript is a derived account.

## Question and information boundaries

The study asks whether semantic-pair calibration changes canonical answer
correspondence beyond giving the model the same transformed inputs separately.
Exact arithmetic, Boolean-assignment and directed-graph oracles distinguish
jointly correct pairs from consistently wrong agreement.

Calibration, development, pilot, final and structural-holdout bases are
disjoint under the documented exact-structure identity. Entity renaming and
fact order are absent from gradient updates but are used in development
selection. Compositions are withheld from both fitting and selection.
The latter are the fully withheld transformation-composition evaluation;
the study does not establish unseen primitive-family selection.

## Prospective and implementation records

- Protocol recorded before pilot inference: d2f70df.
- Deterministic uniqueness amendment before inference: 9994ac0.
- KernelQuant interpretation/control amendment before inference: c37ca8d.
- Evaluated pilot implementation: 431723b.
- Calibration objectives, update budget and selection rule fixed before
  development fitting: a324f7d.
- Corrected calibration row alignment: e6d9174.
- Phased execution and canonical endpoint implementation: 0c3cf72.
- Both development selections, secondary controls, and final analysis frozen
  before final inference: f67baba.

The displayed-A control was added after the pilot. The four-logit mean-drift
diagnostic was added after pilot/Qwen development but before final inference,
without changing the primary scale-calibration comparison. Its parameters
use calibration only. The exact changes and timing are retained in
[calibration-plan.md](calibration-plan.md).

## Verification before reported endpoints

Independent counterexamples identified an incorrect left-padding index and
artificial ties caused by BF16 softmax rounding. Both were corrected before
pilot predictions. Full-default data generation initially encountered repeated
Boolean structures; deterministic rejection and split-subset invariance were
verified before inference.

A real calibration-cache check found extra transform rows in tensors relative
to the selected-pair metadata. The invalid cache was not used by an optimizer.
Inference and metadata were unified on exactly the same ordered 768 inputs,
with tests for shuffled rows, extra transforms and mixed-split refusal.
The two valid Qwen teacher caches used for initial and phased fits have
exactly equal logits, hidden vectors and token counts.

Tests cover exact oracles, canonical non-self-inverse option maps, split
identity, compositions, zero/tail quantizer groups, unpadded versus batched
scoring, finite JS gradients, frozen parameter sets, native tiny Qwen/Llama
checkpoint gradients, exported-scale equivalence, selection and final-phase
guards. A direct 104-test run passed before final inference.

## Pilot and calibration execution

The [pilot report](pilot-results.md) retains both models and all bit widths.
Qwen W4 broke 64 and repaired 63 pairs, with almost no net consistency change.
Smol W4/W3 increased agreement while accuracy fell; its W3 scoring selected
displayed A on all 704 pilot inputs. These observations triggered the
predefined calibration comparison and motivated the constant-letter control.

Each model completed 11 development candidates and all 15 selected fits:
five objectives at seeds 17, 29 and 43. Selected strengths were:

- Qwen: paired 10, random pair 1, kernel 0.1.
- Smol: paired 10, random pair 0.1, kernel 10.

Qwen has 10,235,904 trainable scale deltas; Smol has 12,582,912.
Both models use identical calibration/development records. Every selected
fit executed 96 updates. Actual inputs, tokens, order and optimization
parameters are recorded in fit manifests.

Paired/augmented optimization-time ratios at seeds 17/29/43 were
0.9473/0.9516/1.0032 for Qwen and 1.0072/1.0067/1.0042 for Smol.
No seed crossed the prospective 10% overhead trigger. These measurements
are a budget diagnostic, not an inference-speed benchmark.

## Independent evaluation

The evaluation source is frozen at f67baba. Both model selections and
calibration-only diagnostic offsets were complete before final model
inference. Both models completed all 21 final conditions: four BF16/RTN
conditions, fifteen selected scale fits, and two secondary controls.
Each condition evaluated the same 1,152 bases and 6,144 inputs; the two
models reuse these problems. There are 129,024 prediction rows per model,
258,048 in total. Rows are not independent evaluation problems.
No final outcome selected a setting or checkpoint.

The [final report](final-results.md) retains positive and negative results.
Paired calibration lowered held-out-single consistency for every fit in
both models. Qwen gained joint correctness and composition consistency;
Smol's correctness and composition effects changed sign across seeds.
Both moved away from BF16 predictions. The result does not support a
general advantage for the paired scale recipe.

CPU replay of released predictions reproduced all 21 condition summaries,
36 primary comparisons and 36 secondary composition comparisons per model,
including saved intervals, at absolute tolerance 1e-12. The final offline
suite passed 106 tests. A replay ordering bug affected only descriptive
condition-list order and was fixed without changing numerical evaluation.

The [reproduction guide](reproduce.md) distinguishes model reruns from CPU
replay of saved predictions. Upstream weights and full-vocabulary teacher
caches are not part of the released data.
