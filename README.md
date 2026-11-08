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
storage, deployment-parity, or inference-speed claim.

No quantization calibration or training is implemented.

## Configuration

`configs/pilot.json` pins `Qwen/Qwen2.5-1.5B-Instruct` to an immutable revision
and defines split seeds, split sizes, group size, bit widths, scoring limits,
and bootstrap settings. A non-empty model revision is required.

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

## Tests

Standard library `unittest`. Data-layer tests use only the standard library;
quantizer and scoring tests use tiny CPU tensors and a fake model. No model
downloads or GPU runs are needed.

```bash
python -m unittest discover -s tests -v
```

Tiny-tensor quantizer and fake-model scoring tests run on CPU.
