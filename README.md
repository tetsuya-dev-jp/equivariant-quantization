# eq_quant

Pilot code for a controlled study of whether weight quantization changes the
correspondence between answers to exact semantics-preserving input
transformations.

## Implemented

- Deterministic procedural four-option tasks in three families:
  - `arithmetic`: a named character evaluates a small integer expression.
  - `boolean`: exactly one of four complete variable assignments makes the
    expression true.
  - `graph`: exactly one of four candidate nodes is reachable from a source, or
    lies at a specified directed shortest distance.
- Exact answer oracles recompute every correctness label from the saved
  semantic structure.
- Presentation transforms per base instance:
  - `original`
  - seeded nonidentity `option_permutation`
  - `entity_rename` where meaningful
  - `order_permutation` over independent variable facts or graph edges
- Unsupported family/transform pairs are absent rather than represented as
  identity transforms.
- Deterministic candidate rejection until the requested number of unique base
  structures is obtained. Candidate, rejection, and acceptance counts are saved
  per split and family. A stored duplicate remains an error.
- Exact structural identity excludes distractor choices and the arithmetic
  character name. The boundary is the normalized expression tree, or the graph
  node set, directed edges, source, and relation. It is not full Boolean logical
  equivalence or graph isomorphism.
- All splits are generated in fixed order and then filtered, so building one
  split reproduces the same instances as building the full dataset.
- Structural holdout generation: longer arithmetic expressions, five-variable
  Boolean expressions, and larger directed graphs.
- Hugging Face chat-template next-token scoring for option labels `A`-`D` under
  full precision and symmetric signed RTN W8/W4/W3. Scoring uses left padding,
  attention-derived position ids, and `logits_to_keep=1`.
- Strict greedy-generation parsing: only a bare option letter with limited
  punctuation is accepted; prose is retained as an explicit parse failure.
- RTN groups along input columns with group size 128, including tail groups.
  Linear weights are quantized; `lm_head` and embeddings are excluded.
- Paired, base-cluster bootstrap analysis with JSON and CSV outputs.

The dequantized weights are used for BF16/FP16 arithmetic. This is a numerical
weight-only simulation of quantization error. It makes no packed-integer
storage, deployment-parity, or inference-speed claim. The kernel-loss condition
is a controlled representation-kernel adaptation, not a reproduction of full
KernelQuant or KPQ. Standard AWQ/GPTQ packages are absent from this ROCm
runtime, so this is not a full stronger-PTQ benchmark and RTN is not claimed as
state of the art.

Calibration keeps the original RTN integer codes, biases, embeddings, output
head, and all pretrained parameters frozen. It learns one positive scale per
output row and input group, parameterized by its natural logarithm and
initialized to the RTN scale. Forward passes reconstruct BF16 weights from the
frozen codes; zero RTN groups stay exactly zero. Five objectives are available:
original-only KD, augmented KD, paired canonical-option JS, a random-pair JS
control with a within-batch nonidentity shift, and a representation-kernel
control using cosine Gram matrices of the final normalized hidden vectors.

## Configuration

`configs/pilot.json` pins `Qwen/Qwen2.5-1.5B-Instruct` to an immutable revision
and defines split seeds, split sizes, group size, bit widths, scoring limits,
and bootstrap settings. A non-empty model revision is required.

`configs/calibration.json` combines those pinned pilot fields with the frozen
calibration defaults: W4, group size 128, AdamW with learning rate 0.001,
betas (0.9, 0.999), zero weight decay, gradient clipping 1.0, four bases
(eight prompts) per update, 96 updates over the 384 calibration bases, and
temperature 1. The frozen fit seeds are 17, 29, and 43. The frozen
regularization candidates are 0.1, 1, and 10, but the weight is never selected
implicitly: pass the explicit `--lambda` value for `paired`, `random_pair`, and
`kernel`.

## Commands

Build the pilot split (64 base instances per family by default):

```bash
python -m eq_quant data build \
  --config configs/pilot.json \
  --output local/data/pilot \
  --split pilot
```

Build all declared splits:

```bash
python -m eq_quant data build \
  --config configs/pilot.json \
  --output local/data/all \
  --split all
```

Score the records under FP and then RTN W8/W4/W3:

```bash
python -m eq_quant evaluate \
  --config configs/pilot.json \
  --records local/data/pilot/records.jsonl \
  --output local/results/pilot
```

Evaluation refuses to overwrite existing `run.json` or `predictions.jsonl`
unless `--overwrite` is passed. Each completed condition is persisted in its
own `predictions_<condition>.jsonl` and appended to `predictions.jsonl`.
`run.json` records `running`, `completed`, or `failed` state and the completed
conditions.

Add an optional greedy raw-generation subset per condition:

```bash
python -m eq_quant evaluate \
  --config configs/pilot.json \
  --records local/data/pilot/records.jsonl \
  --output local/results/pilot \
  --generation-subset 32
```

Analyze paired predictions:

```bash
python -m eq_quant analyze \
  --config configs/pilot.json \
  --predictions local/results/pilot/predictions.jsonl \
  --output local/results/pilot/analysis
```

### Calibration

Build the calibration-only records used by the fit:

```bash
python -m eq_quant data build \
  --config configs/calibration.json \
  --output local/data/calibration \
  --split calibration
```

Cache the complete final next-token teacher logits and final normalized hidden
vectors over calibration `original` and `option_permutation` records only. A
mixed-split records file is filtered to calibration records; the `calibrate`
command itself refuses any non-calibration record. The cache records the model
id/revision/dtype, the scoring limits, and the exact ordered record metadata
(record id, base id, task family, transform, prompt, and option order).

```bash
python -m eq_quant cache-teacher \
  --config configs/calibration.json \
  --records local/data/all/records.jsonl \
  --output local/calibration/teacher-cache
```

Run one objective for one explicit lambda and one fit seed. Regularized
objectives require `--lambda`; the two KD controls refuse it. The command
refuses records outside the calibration split (including a configured split
other than `calibration`) and refuses a cache whose ordered record metadata,
model revision/dtype, or scoring limits do not match.

```bash
python -m eq_quant calibrate \
  --config configs/calibration.json \
  --records local/data/calibration/records.jsonl \
  --teacher-cache local/calibration/teacher-cache \
  --output local/calibration/fits/paired-w4-s17 \
  --objective paired \
  --lambda 1.0 \
  --seed 17
```

Available objectives are `original_kd`, `augmented_kd`, `paired`,
`random_pair`, and `kernel`. Run the lambda search and seed replication as
separate commands; no hyperparameter or checkpoint is selected inside a fit.

Score arbitrary supplied records with exported scales applied to the pinned
original RTN integer codes. The shared scorer and condition name are explicit.

```bash
python -m eq_quant evaluate-calibrated \
  --config configs/calibration.json \
  --records local/data/development/records.jsonl \
  --scales local/calibration/fits/paired-w4-s17 \
  --output local/results/development-paired-w4-s17 \
  --condition paired_w4_s17
```

Both `calibrate` and `evaluate-calibrated` refuse to overwrite existing output
unless `--overwrite` is passed. Calibration fits do not read final outcomes and
do not make selection decisions.

## Data record fields

Each record includes:

- `record_id`, `base_id`, `split`, `task_family`, `transform`
- `semantic` (original structure and canonical option identities)
- `prompt`
- `option_order` (for each displayed slot, the canonical option index)
- `correct_option_index` (displayed slot)
- `rename` and `order` transform metadata when applicable

## Outputs

- `bases.jsonl`: deterministic original structures and semantic keys.
- `records.jsonl`: rendered records.
- `manifest.json`: generation settings and counts.
- `predictions.jsonl` and `predictions_<condition>.jsonl`: per-record labels,
  option logits, display and canonical probabilities, predictions, token counts,
  and semantic identity.
- `run.json`: execution state, model revision, software/hardware,
  per-condition timing, peak GPU allocation, module inventory, quantized code
  ranges, and zero-group counts.
- `analysis.json`, `summary.csv`, `comparisons.csv`, `bootstrap.csv`: pooled
  and per-transformation summaries, raw counts, base-macro rates, micro rates,
  union-bound diagnostics, transition counts, and base-cluster bootstrap
  intervals. Unavailable family/transform combinations are emitted explicitly
  with `available=false`.
- `teacher_cache.pt` and `cache_manifest.json`: complete last-token vocabulary
  logits and final normalized hidden vectors, keyed by model revision and
  ordered record metadata (record id, base, prompt, option order), plus scoring
  limits, token counts, and runtime provenance.
- `calibration.json` and `scales.pt`: fit state (`completed` or `failed`),
  complete effective config, objective, explicit lambda, seed, model revision,
  quantization description, optimizer and schedule, trainable parameter count,
  per-update KD/regularizer losses and gradient norms, exact nonpadding and
  padded token counts, batch order, timing, peak memory, and detached FP32 CPU
  relative log scale deltas. The effective scale is the fixed FP32 RTN scale
  times `exp(delta)`, so delta zero reproduces the RTN BF16 weights exactly.
- Calibrated evaluation `run.json` and `predictions.jsonl`: explicit condition
  name, scale source and fit metadata, model provenance, timing, peak memory,
  and the shared scorer's per-record predictions.

## Tests

Standard library `unittest`. Data-layer tests use only the standard library;
quantizer, scoring, and calibration tests use tiny CPU tensors and fake models.
No model downloads or GPU runs are needed. Calibration tests cover exact RTN
initialization and zero/tail groups, exported-delta reproduction on real tiny
Qwen2 and Llama layers, exercised non-reentrant gradient checkpointing with
early/late scale gradients, the shared scorer, objective semantics including
finite JS gradients for degenerate distributions, record-metadata cache
refusal, split refusal, and overwrite protection.

```bash
python -m unittest discover -s tests -v
```

Tiny-tensor quantizer and fake-model scoring tests run on CPU.
