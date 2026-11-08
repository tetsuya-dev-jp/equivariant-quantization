# Prior-work and novelty audit

Searched on 8 November 2026. This is a bounded literature search, not proof
that no overlapping work exists. No empirical novelty or benefit is established.

## Closest verified sources

1. **FidelityQ: Preserving Predictive Fidelity in Ultra-Low-Bit Language Model
   Quantization**, Chen et al., arXiv:2610.14934v1 (10 October 2026).
   https://arxiv.org/html/2610.14934
   Read abstract, method outline and section 3. KIC selects sequences using
   teacher/student token divergence and optimizes reconstruction levels with
   frozen pretrained weights. This is a mandatory comparison in interpretation:
   preserving prediction fidelity and tuning quantization scales are not new.
   The retrieved method does not define canonical answer mappings between
   semantics-preserving input pairs.

2. **Target-Aware Calibration Data Selection for Preserving Uncertainty in
   Quantized Language Models**, Yang et al., arXiv:2608.21019v1
   (21 August 2026). https://arxiv.org/html/2608.21019
   Read abstract and sections 3-4. DPQ constructs doubt/anchor mixtures and
   evaluates option-distribution and decision-boundary preservation. Targeting
   behavior beyond average accuracy is already established. Our question is
   relational preservation under exact transformations, with matched exposure.

3. **OSTQuant: Refining Large Language Model Quantization with Orthogonal and
   Scaling Transformations for Better Distribution Fitting**, Hu et al.,
   arXiv:2501.13987. https://arxiv.org/html/2501.13987
   Read abstract and method sections. Learned equivalent transformations and
   KL-Top teacher supervision optimize quantization. End-to-end output matching
   is a prior baseline, not a proposed contribution here. Internal equivalent
   weight transformations differ from semantic transformations of user input.

4. **FPTQuant: Function-Preserving Transforms for LLM Quantization**,
   van Breugel et al., arXiv:2506.04985v2 (8 July 2026), ICML 2026 according
   to the arXiv record. https://arxiv.org/abs/2506.04985
   Read the record and abstract. This uses internal transformer symmetries to
   improve quantization distributions and matches FP/quantized outputs. Its
   use of equivariance does not itself establish a semantic-input experiment.

5. **Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of
   Quantized LLMs**, Fu et al., EMNLP 2025, pages 30435-30458.
   https://aclanthology.org/2025.emnlp-main.1548/
   Read the abstract and prompt-sensitivity section during topic selection.
   Quantization/prompt sensitivity is known. A new study cannot claim its
   first observation; exact executable transformations and a matched objective
   comparison are the narrower proposed contribution.

6. **When Quantization Preserves Accuracy but Not Evidence: Explanation-Aware
   Post-Training Quantization for Medical LLMs**, Kim, Kim and Goebel,
   arXiv:2609.24799v1 (21 September 2026).
   https://arxiv.org/html/2609.24799
   Read abstract and introduction/method outline. It adds teacher-rationale
   preservation losses to transformation-based PTQ. Behavior-aware calibration
   losses and accuracy-versus-behavior dissociation are therefore not new.
   It targets rationale/evidence relations rather than exact input orbits.

7. **Does Quantization Change What Agreement Means? A Paired Audit of
   Homogeneous LLM Panels**, Bala, UncertaiNLP 2026.
   https://aclanthology.org/2026.uncertainlp-main.37/
   Abstract retrieved in search; full methods not yet inspected. The reported
   setting compares stochastic panels on identical prompts, rather than exact
   canonicalized input transformations. Keep this distinction provisional.

## Searches and candidate claim

Web searches covered quantization + semantic equivariance, paired consistency,
metamorphic tests, paraphrase/calibration objectives and prompt robustness.
The preceding topic-selection turn additionally used alphaXiv and Exa for
direct counterexamples. This run checked newer fidelity/uncertainty and
explanation-preserving objectives to avoid an overly broad novelty claim.

Candidate contribution: an oracle-verified, split-controlled assessment of
quantization-induced changes in canonical answer correspondence, followed
conditionally by a same-data/same-update paired-calibration comparison on
held-out transformations. Exact agreement alone is not correctness; unchanged
average accuracy alone is not pointwise fidelity. New pair inconsistency
necessarily entails at least one changed pointwise prediction, so it is not
an independent failure mechanism proved merely by a consistency score.

No direct same-objective/same-evaluation match was found in the retrieved
sources. This supports investigating the question, not claiming priority or
superiority. Recheck related work before public manuscript finalization.
