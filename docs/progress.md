# Work record

## Objective and acceptance

Complete the prospective protocol, conditional calibration study, reproducible
artifacts, manuscript draft, and public GitHub publication. Use DeepSeek V4.1
Flash for implementation. Preserve the full goal across sessions.

## Verified on 8 November 2026

- New independent project at `/home/tetsuya/dev/equivariant-quantization`.
- AMD Radeon RX 9070 XT, about 16 GiB, usable through ROCm PyTorch.
- Research environment: PyTorch 2.12.0+rocm7.14.0, Transformers 5.14.1.
- OpenCode exposes `opencode-go/deepseek-v4.1-flash`.
- GitHub CLI authenticated as `tetsuya-dev-jp`; public publication authorized.
- FidelityQ and DPQ are close prior work; the proposed difference concerns
  exact input transformations rather than ordinary prediction agreement.
- Prospective protocol committed before empirical evaluation: `d2f70df`.
- Isolated local runtime reuses ROCm PyTorch 2.12.0 and Transformers 4.57.1;
  the separately inspected research environment had Transformers 5.14.1.
- Pinned primary model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Primary-model tokenizer represents A/B/C/D as tokens 32/33/34/35.
- A one-question GPU infrastructure smoke returned the expected B for 2+2;
  peak allocated memory was 2.958 GiB. This is not a research endpoint.

## Current milestone

Calibration development after both completed pilots. Final outcomes remain
unexposed. The frozen next-stage specification is `docs/calibration-plan.md`.

DeepSeek implementation uses OpenCode session `ses_e46df35a1ffeXRWuz0zIqZY72D`.
The initial high-reasoning CLI (handle 92657) was intentionally interrupted
after files and 20 tests were written, to redirect it to confirmed review
findings. Its process exited 130. Correction pass 2 (handle 83995) was also
interrupted after its corrected files passed 28 direct offline tests. A fresh
DeepSeek session `ses_e46c0ce95ffejrZgGvrq1LwjMC` (handle 56417) completed
the independently found BF16 probability-rounding correction. Task/events/session
exports are private under `local/delegation/` and must not be published.
The SmolLM2-1.7B download (handle 33747) completed successfully at revision
`31b70e2e869a7173562077fd711b654946d38674`.
Check live handles/current files before resuming; do not blindly restart.

## Review evidence before research endpoints

- Initial 20 offline tests pass in the isolated environment, but they do not
  prove correctness: the left-padding test duplicated an erroneous index.
- Independent counterexample: last-token indices yielded [2,9] instead of
  [4,9] for unequal-length left-padded sequences. Must be corrected.
- Full-default data generation initially failed on duplicate Boolean bases.
  Prospective amendment 1 records the deterministic uniqueness fix; final
  structural keys must also exclude distractors and preserve subset generation.
- Review also identified ambiguous graph-distance wording, a permissive
  generation parser, unequal base weighting, and blank CSV fields. Findings
  are batched in `local/delegation/01-review.txt` for the correction pass.
- These checks preceded all pilot predictions. Final holdout outcomes remain
  unexposed.
- Corrected default generation, subset invariance, canonical option transport,
  base-macro aggregation, output persistence, and independent padding tests
  passed in a direct 28-test run. The pilot dataset has 192 bases / 704 records.
- A native tiny random Qwen model with the actual tokenizer gave matching
  predictions when unequal-length prompts were scored in a batch or alone;
  the maximum option-logit difference was 1.49e-8. This is an infrastructure
  check, not a pretrained-model research endpoint.
- BF16 logits [0, 0.001, 0, 0] yield four equal BF16 softmax probabilities,
  changing an argmax from index 1 to index 0. The final correction requires
  FP32 probabilities and predictions from logits, plus a regression test.

## Completed pilot evidence

Source revision `431723b`; all 30 tests passed directly before execution.
Each model evaluated 192 bases / 704 records / 512 original-variant pairs.
Raw predictions and run metadata are private under `local/results/qwen-pilot`
and `local/results/smol-pilot` until the curated public results are assembled.
Qwen execution handle 83240 and Smol handle 51647 completed successfully.
Both models use BF16 arithmetic and dequantized RTN weights, not packed kernels.

- Qwen accuracy (base macro), FP/W8/W4/W3: 0.506076, 0.501302,
  0.482639, 0.243490; consistency: 0.668403, 0.681424, 0.664931, 0.473958.
- Qwen W4 broke 64 and repaired 63 pairs. Its consistency difference was
  -0.003472 (base-bootstrap 95% interval [-0.053841, 0.045139]). W3 difference
  was -0.194444 ([-0.255273, -0.134505]). This triggers calibration by the
  predefined broken-pair count; it does not show a net W4 consistency decline.
- Smol accuracy FP/W8/W4/W3: 0.400174, 0.398003, 0.265191, 0.229601;
  consistency: 0.603299, 0.591146, 0.689236, 0.704861. W4 broke 54 and
  repaired 103 pairs. Increased agreement coexists with lower correctness.
- Bootstrap: 1000 resamples of base problems, seed 4108; raw counts and
  all task/transform outcomes are retained. Pilot findings are exploratory.

## Active bounded implementation

- Calibration API/CLI: initial DeepSeek session `ses_e46bbfb84ffe7EN1UyGBhV9tt7`,
  handle 9266, was interrupted after 13 tiny tests passed to redirect it to
  confirmed review findings. Correction session `ses_e46b21163ffed6Z17d01HhvDgk`,
  handle 77795, is active; task `local/delegation/04-review.txt`, events
  `09-events.jsonl`. Correct checkpointing activation, stable RTN-relative log
  scales, finite JS gradients, direct cache input matching without hashes,
  strict calibration boundary, and native tiny HF integration before running.
- Final-only exact compositions: session `ses_e46bbf6c4ffexx242Sxy0GFstE`,
  handle 29166, completed, events `05-events.jsonl`. Split-aware analysis
  availability follow-up (handle 77100, session `ses_e46b4ef58ffeHGZrur0CB3Hrtb`)
  completed; 20 direct data+analysis tests passed. Full dataset is now 9,664
  records: original 7,744 plus 1,920 final-only compositions. Existing records
  are unchanged.
- Explicit analysis reference: session `ses_e46bbbf00ffeFRaamtsPYoKZTN`,
  handle 10293, completed; five direct analysis tests passed. Changes reviewed,
  not committed yet. Events `06-events.jsonl`.
- Endpoint/selection helper: session `ses_e46b37f54ffePj9HFL6OEzpiyE`, handle
  61520, completed; 13 direct tests passed, events `08-events.jsonl`. Review
  found comparison currently intersects missing support rather than requiring
  exact support as requested; correct this before final comparisons.
- Pilot plots: session `ses_e46b0c1f5ffebwCXnw89BMC0lS`, handle 46727,
  completed; `scripts/plot_pilot.py`, five exports in `figures/pilot`. Both
  PNGs visually inspected and table values checked against saved analyses.
  Curated pilot data/results exist at `data/pilot` and `results/pilot`, still
  unpublished. English report `docs/pilot-results.md` references both figures.
- Phased study runner: handle 6029, events `11-events.jsonl`, active; owned
  files `eq_quant/study.py` and `tests/test_study.py`. It must complete develop
  selection before final, and report the >10% time-control requirement.
Independent workers have disjoint file ownership; check state before resuming.

The Smol W3 pilot selected displayed A on all 704 records. Added a prospective
fixed-displayed-A final negative control in calibration-plan.md. Ensure the
study runner/report implements it; it was not in the original runner prompt.
No actual calibration/teacher-cache execution has happened yet. No public
GitHub repository has been created yet.

## Remaining

Calibration implementation review and provenance; matched development fits and
selection; three-seed calibration; independent final evaluation;
figures and English manuscript; publication and clean-clone verification.
