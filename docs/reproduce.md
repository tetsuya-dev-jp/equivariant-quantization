# Reproducing the study

Use Python 3.12 and a PyTorch build appropriate for the machine. The observed
experiment runtime is PyTorch 2.12.0+rocm7.14.0 on a Radeon RX 9070 XT,
Transformers 4.57.1, and BF16 arithmetic. The remaining tested package versions
are in [requirements-reproduce.txt](../requirements-reproduce.txt).
This file deliberately does not replace a hardware-specific PyTorch build.

Start from a fresh checkout and environment:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
~~~

Install PyTorch for the intended platform using its
[official installation instructions](https://pytorch.org/get-started/locally/),
then install the pinned non-Torch dependencies and this package:

~~~bash
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
~~~

For CPU-only tests, the official Transformers 4.57.1
[installation guide](https://huggingface.co/docs/transformers/v4.57.1/installation)
also documents a CPU PyTorch index. Tests use tiny random/fake models and
synthetic data; downloaded model weights and a GPU are not required.
The reported large-model runs used the observed ROCm environment; other
hardware/library combinations have not been validated as equivalent.

## Replay released predictions without model weights

Recompute the frozen summaries and paired base-bootstrap intervals directly:

~~~bash
python scripts/replay_final.py --source results/final/qwen --output local/replay/qwen
python scripts/replay_final.py --source results/final/smol --output local/replay/smol
~~~

The replay reads gzip-compressed per-input predictions, recomputes correctness
from the oracle, uses the saved condition/seed and bootstrap settings, and
checks the released metrics with an absolute floating-point tolerance of
1e-12. It writes regenerated results and a verification report into a fresh
output directory. It does not load pretrained models or scale checkpoints.

Recreate the figures and evidence tables:

~~~bash
python scripts/plot_pilot.py
python scripts/plot_final.py
~~~

The plots are generated from saved numbers. PDF exports are vector graphics;
PNG exports support preview. The primary effect figure retains each fit seed,
and outcome stacks are descriptive means rather than independent seed tests.

## Rerun the model experiments

The two pinned study configurations are
[calibration.json](../configs/calibration.json) and
[smol-calibration.json](../configs/smol-calibration.json).
Each development phase performs two KD controls plus three regularized
objectives at three candidate strengths. Selection uses development only.
Run the two developments before any final inference:

~~~bash
python -m eq_quant.study --config configs/calibration.json --root local/reproduction/qwen --phase develop
python -m eq_quant.study --config configs/smol-calibration.json --root local/reproduction/smol --phase develop
~~~

Fit the selected settings with the other two seeds, and construct the
calibration-only four-logit diagnostic:

~~~bash
python -m eq_quant.study --config configs/calibration.json --root local/reproduction/qwen --phase replicate
python -m eq_quant.study --config configs/smol-calibration.json --root local/reproduction/smol --phase replicate
python -m eq_quant.study --config configs/calibration.json --root local/reproduction/qwen --phase diagnostic
python -m eq_quant.study --config configs/smol-calibration.json --root local/reproduction/smol --phase diagnostic
~~~

Check the recorded paired/augmented optimization-time ratios before making a
matched-compute claim. The prospective rule requires an additional time-matched
augmented control if overhead exceeds 10%. This is separate from equal update,
input, and parameter counts.

Final inference requires frozen selection, every selected fit, and compatible
diagnostic offsets. Run it only after these prerequisites are complete:

~~~bash
python -m eq_quant.study --config configs/calibration.json --root local/reproduction/qwen --phase final
python -m eq_quant.study --config configs/smol-calibration.json --root local/reproduction/smol --phase final
python -m eq_quant.study --config configs/calibration.json --root local/reproduction/qwen --phase report
python -m eq_quant.study --config configs/smol-calibration.json --root local/reproduction/smol --phase report
~~~

Use one GPU job at a time for the timing comparison. Compatible completed
outputs can be reused. Existing incomplete or incompatible output is refused;
inspect a failed run and preserve it or choose a fresh root rather than
silently overwriting it.

The default generator has 2,112 base problems and 9,664 rendered records:
1,408 calibration, 1,408 development, 704 pilot, 4,096 final, and 2,048
structural-holdout records. Only originals and option permutations are used
for fitting; renaming/order and final compositions remain evaluation inputs.
See the [protocol](protocol.md), [calibration plan](calibration-plan.md), and
[prior-work audit](related-work.md) for the boundaries and amendments.

## Build the manuscript

From the repository root:

~~~bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../local/paper-build draft.tex
~~~

The manuscript source is English LaTeX. The released draft includes explicit
scope limits; it is not an arXiv submission or a claim of peer review.
