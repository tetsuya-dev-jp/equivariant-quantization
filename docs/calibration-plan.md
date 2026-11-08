# Calibration plan, frozen after the pilot and before development

The Qwen pilot at source revision `431723b` triggered calibration: W4 broke
64 previously consistent pairs and repaired 63 of 512 pairs; W3 broke 166.
The W4 base-macro consistency change was -0.0034722, not a general decline.
The pilot is development evidence, not a final test of the method.

## Model and objectives

Use W4, group size 128, the same signed RTN codes as the pilot. Freeze codes,
biases, embeddings, output head, and all pretrained parameters. Learn positive
per-output-row/per-input-group scales, parameterized by their logarithms and
initialized to RTN scales. Keep all-zero groups identically zero. Reconstruct
BF16 weights during forward passes; retain FP32 scale parameters and losses.
Do not add low-rank updates or alter integer codes. Report the trainable count.

Cache the BF16 teacher's complete final next-token logits and final normalized
hidden vector on calibration inputs. No teacher answers or oracle labels are
used as training targets. The shared distillation loss is full-vocabulary
KL(teacher || student), temperature 1, averaged over prompts.

Fit five controls from identical initial scales:

1. Original-only KD: duplicate each original instead of its option-permuted
   partner, matching prompt count and updates; report actual token exposure.
2. Augmented KD: original and its option-permuted partner, independently.
3. Paired: augmented KD plus lambda times Jensen-Shannon divergence between
   the two four-option distributions after mapping to canonical identities.
4. Random-pair: the same augmented KD plus the same JS loss, but match each
   original to another base's variant using a within-batch nonidentity shift.
5. Kernel: augmented KD plus lambda times mean squared difference between
   teacher and student cosine Gram matrices of the final hidden vectors in
   the same minibatch. This is a controlled representation-kernel adaptation
   motivated by KernelQuant, not a reproduction of full KernelQuant or KPQ.

The paired objective imposes invariance, which may conflict with an inconsistent
teacher. It does not simply preserve every teacher relation. Evaluate teacher
prediction fidelity, accuracy, jointly correct and consistently wrong pairs.

## Optimization and selection

Use AdamW, learning rate 0.001, betas (0.9, 0.999), weight decay 0, gradient
norm clipping 1. Each minibatch contains four bases / eight prompts. Shuffle
384 calibration bases once and take 96 updates: one epoch, 768 prompts.
Use fit seeds 17, 29, 43. No dropout. Do not select intermediate checkpoints.
Record losses, input/nonpadding token counts, elapsed GPU time, peak memory,
and the complete configuration. All losses are FP32. Use identical batch
orders and parameter initializations across objectives within each seed.

For each regularized objective, search lambda in {0.1, 1, 10} using seed 17
and development only. Choose the candidate with the highest base-macro joint
correctness on held-out entity renaming / fact order, subject to overall
development accuracy at least augmented KD minus one percentage point.
Break ties by held-out consistency, then smaller lambda. If no candidate
meets the accuracy condition, choose the highest overall development accuracy,
then joint correctness, consistency, and smaller lambda. Retain all candidate
outcomes. The two KD controls have no lambda search. Reuse the chosen seed-17
fit and train seeds 29 and 43. Apply this selection rule separately per model.

The primary comparison remains paired versus augmented KD at W4 on held-out
transformations. Report each fit seed and paired base-bootstrap intervals;
do not count seeds as new test problems. Also report original/option changes,
structural holdout, and compositions. No final outcome may select a setting.
Compare optimization time for equal updates; if paired overhead exceeds 10%,
add an augmented-KD run with extra updates matching its measured time.

## Final data and scope

Add option+rename and, where available, option+rename+order compositions only
to final and structural holdout. Reuse each component's exact saved map;
compose renderings without changing canonical options or oracle structures.
This adds evaluation records, not new base problems, and leaves existing
pilot/calibration/development records unchanged.

Replicate on SmolLM2-1.7B-Instruct at revision
`31b70e2e869a7173562077fd711b654946d38674` using independent development
selection and the same final questions. This study does not claim an optimized
deployment format. Standard AWQ/GPTQ packages are absent from this ROCm runtime;
the kernel-loss control is not a substitute for a full stronger PTQ benchmark.
Record this comparison limitation explicitly rather than calling RTN SOTA.

Implementation is delegated to DeepSeek V4.1 Flash. The supervisor reviews
tests and executes the actual model runs. No implementation worker may inspect
final model outcomes or make selection decisions from them.
