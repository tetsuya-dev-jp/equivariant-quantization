# Protocol: transformation-preserving language-model quantization

Status: prospective protocol, 8 November 2026. No empirical benefit established.

## Question and scope

Does weight quantization alter the correspondence between answers to exact
semantics-preserving input transformations? Can paired calibration preserve
that correspondence beyond adding the same transformed examples individually?
This study concerns controlled multiple-choice reasoning, not general reasoning
or inference acceleration. Implementation is delegated to DeepSeek V4.1 Flash;
Codex designs, reviews, executes, and verifies the study.

## Data and information boundaries

Use procedurally generated four-option arithmetic, Boolean evaluation, and
directed-graph reasoning questions with executable exact answer oracles.
All options must have distinct meanings/values and exactly one correct answer.
Each instance has an explicit canonical answer identity and option permutation.
Use base semantic structures disjoint across calibration, development, pilot,
and final evaluation. Variants of one base instance always stay in one split.
Generation must fail on accidental cross-split semantic collisions, not silently
drop or reuse them. Save generation settings and full per-item metadata.

Pilot: 64 independent base instances per family, 192 total, each with original,
option permutation, entity renaming where meaningful, and sentence/order
transformation where meaningful. No transformation may alter semantics by
changing arithmetic order or graph direction. Unsupported family-transform
combinations are explicitly unavailable, not identity transformations.

Calibration and development: 128 base instances per family for each split.
Final evaluation: 256 fresh base instances per family plus a separate 128-base
per-family structural holdout (longer expressions or unseen graph sizes).
Pilot exposure is recorded; final evaluation seeds and structures are not used
for selecting prompts, clipping, calibration hyperparameters, or checkpoints.

Train/calibrate on option permutations only. Entity renaming and sentence/order
changes are held-out transformation families; the final dataset also contains
compositions of individually defined transformations. Use fixed deterministic
split seeds (calibration 1108, development 2108, pilot 3108, final 4108,
structural holdout 5108); record any justified amendment before its execution.

## Models, scoring, and quantization

Start with Qwen/Qwen2.5-1.5B-Instruct at a pinned revision; replicate with a
different model family if the pilot is informative. Fix the official chat
template, English prompt, answer instructions, and context budget across
conditions. No answer labels appear in inference prompts.

Primary endpoint: next-token logits for four verified single-token option
labels, mapped back to canonical option identities before comparisons.
This is constrained multiple-choice scoring. Greedy free generation on a
predefined subset is an additional endpoint, with raw output and parsing
failures retained; do not conflate the endpoints.

Full-precision baseline is explicitly BF16/FP16 as actually executed. Quantize
all transformer linear weights except token embeddings and output head with
a specified group size, symmetric signed integer quantizer, and saved scales.
Begin with group size 128, round-to-nearest, W8/W4/W3, activations BF16/FP16.
Record exact range, clipping, excluded modules and effective storage overhead.
Dequantized arithmetic is a numerical quantization experiment: do not claim
packed-integer memory savings, deployment parity or speed gains from it.
Export packed integers/scales and verify reconstruction equivalence if a
deployable artifact is claimed. Include a standard stronger PTQ baseline in
the expanded comparison when operationally feasible, or document its absence.

## Predefined pilot decision

Report accuracy and consistency changes with paired intervals, including gains
and losses. Advance to paired-calibration development if quantization breaks
at least 10 previously consistent base/variant pairs across the pilot, or
increases canonical-answer disagreement by at least two percentage points
for W4 or W3. This is a development trigger, not a significance claim.
If the trigger fails, expand the diagnostic once to a second family and/or
larger model before deciding the phenomenon is unsupported in this scope.
Negative results still require the predefined final evaluation and report.

## Conditional calibration comparison

If triggered, update quantizer clipping/scales only, leaving pretrained weights
frozen. Compare uncalibrated RTN; unpaired teacher-output distillation on base
examples; unpaired distillation on the same base+variant tokens; and the latter
with canonicalized paired consistency regularization. A random-pair control
tests whether correspondence matters. Pair consistency alone cannot establish
correctness: retain teacher fidelity and report jointly correct pairs.

Every optimized comparison uses the same initialization, update count, batch
size, data/token exposure, optimizer, and trainable parameters. Account for
extra paired-loss compute separately; include a time/compute-matched control
if the proposed method consumes materially more work. Hyperparameters and
checkpoint selection depend only on development. Use three calibration seeds.

## Metrics, uncertainty, and completion

Save per-item labels, canonical predictions, four logits/probabilities,
correctness, transformation map, lengths and execution provenance.
Report original and transformed accuracy, canonical consistency, jointly
correct pairs, consistent-but-wrong pairs, FP-to-quantized answer flips,
and consistent-to-inconsistent versus reverse transitions. Avoid interpreting
agreement alone as improved reasoning or attributing drift solely to accuracy.
Resampling unit is the base problem, carrying all transformations with it;
report calibration seeds separately. Primary comparison is paired calibration
versus unpaired augmentation at W4 on final held-out transformations.

Deliver executable oracle/quantizer/alignment tests, pinned configurations,
raw predictions, environment and timing records, reproducible summaries and
figures, an evidence-grounded English manuscript draft, limitations and
third-party attribution, and a public GitHub repository. Do not redistribute
upstream weights or secrets. Publication is authorized by the user; arXiv
submission, paid compute purchases, and claims of positive results are not.

## Prospective amendment 1: unique candidate generation

Before model evaluation, a full-default generator check failed because the
Boolean sampler produced duplicate base questions within calibration.
Replace unconditional one-candidate-per-item generation with deterministic
candidate rejection until the predeclared number of unique base structures
is obtained. Save candidate/rejection counts and reasons. A duplicate in the
completed stored dataset remains an error. Exclude distractor options from
structural identity so new distractors cannot evade split disjointness.
The identity boundary is exact normalized problem structure, not arbitrary
Boolean logical equivalence or graph isomorphism. Generating a split alone
must reproduce its instances from full generation. Counts, seeds, task support
and scoring are unchanged. This amendment precedes all research endpoints.
