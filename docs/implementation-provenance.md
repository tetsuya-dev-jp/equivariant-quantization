# Implementation and verification provenance

The experimental implementation was delegated through OpenCode using
opencode-go/deepseek-v4.1-flash. Codex specified the study, reviewed the
implementation, executed the experiments, checked evidence, and drafted the
research documents. The model identity was checked in assistant
message metadata for both implementation and correction sessions; those
checks returned provider opencode-go and model deepseek-v4.1-flash.

The verification used counterexamples as well as tests. A passing test that
repeated a faulty padding formula was replaced by an independent unpadded
reference. A BF16 near-tie example verified prediction from logits rather than
rounded probabilities. A full generated calibration file exposed a teacher
cache alignment bug that two-transform-only fixtures missed. The corrected
cache was tested with shuffled inputs and extra evaluation transforms before
any optimizer fit used it.

Native tiny Qwen2 and Llama instances verified early/late scale gradients,
active non-reentrant checkpointing, and identical exported/reloaded
predictions. The actual GPU fits then recorded exact settings, inputs,
token counts, batch order, losses, parameter names, timing and state.

The calibration source was frozen at 0c3cf72 for the replicated fits.
Independent inference and comparison code was frozen at f67baba before any
final model outcomes. The original and augmented Qwen CLI prefits used the
same committed calibration implementation; regenerated input records and
shared teacher logits, hidden vectors and token counts were explicitly
checked for equality before reuse.

Post-inference changes corrected replay comparison of membership-only condition
lists and wrapped figure labels. Released-data replay verified the numerical
summaries and intervals; direct comparisons verified that the layout change
altered only display labels in plot metadata and left both CSV tables intact.

[environment.json](../environment.json) records the observed package/hardware
versions. [The experiment record](progress.md) and
[reproduction guide](reproduce.md) connect the protocol, selection, source,
results and regenerated analysis.

Private API/session logs are not part of the research data. The release
contains source, synthetic inputs, numerical predictions and evidence
artifacts, with upstream model weights referenced by immutable revisions.
AI tools are disclosed as implementation/drafting assistance, not as authors
or substitutes for the evidence checks.
