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

Freeze protocol, delegate offline dataset/oracle plus model-scoring and RTN
implementation, review meaningful tests, then run the pilot. No results yet.

DeepSeek implementation is running in OpenCode session
`ses_e46df35a1ffeXRWuz0zIqZY72D` (exec handle 92657). Task and events are
under ignored `local/delegation/`. Model download exec handle: 33747.
Check live handles/current files before resuming; do not blindly restart.

## Remaining

Novelty evidence ledger; implementation and provenance; actual pilot; trigger
decision; calibrated or negative-result path; independent final evaluation;
figures and English manuscript; publication and clean-clone verification.
