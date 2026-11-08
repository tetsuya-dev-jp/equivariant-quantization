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
   Publication update verified in this run: Findings of EMNLP 2026,
   pages 14073-14093, https://aclanthology.org/2026.findings-emnlp.670/ .

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
   Publication independently verified: EMNLP 2026, pages 41629-41651,
   https://aclanthology.org/2026.emnlp-main.1828/ .

7. **Does Quantization Change What Agreement Means? A Paired Audit of
   Homogeneous LLM Panels**, Bala, UncertaiNLP 2026.
   https://aclanthology.org/2026.uncertainlp-main.37/
   Official record and abstract read; full methods not yet inspected. The reported
   setting compares stochastic panels on identical prompts, rather than exact
   canonicalized input transformations. Keep this distinction provisional.

8. **KernelQuant: Post-Training Quantization via Representation Kernel
   Preservation**, Park, Kwon, Kim and Lee, EMNLP 2026, pages 19833-19850.
   https://aclanthology.org/2026.emnlp-main.856/
   Official abstract and PDF section 3 (pages 2-4) read in the later
   adversarial pass. It selects
   quantization scales to preserve pairwise cosine kernels among calibration
   inputs, with error compensation, rotation and activation clipping. Therefore
   preserving relations across inputs during PTQ is also prior work; the claim
   cannot be that relational quantization is new. The proposed distinction is
   explicit known semantic transformations, option-identity transport and
   held-out transformation transfer. Its layerwise scale search and Hessian
   compensation differ from an end-to-end paired-answer calibration objective;
   a controlled kernel-preservation comparison is included prospectively.

9. **Semantic Invariance in Agentic AI**, arXiv:2603.13173v2.
   https://arxiv.org/html/2603.13173v2
   Exa retrieved its abstract and formal transformation definitions. It evaluates
   semantic-preserving prompt changes across agents; the retrieved text does
   not describe a PTQ calibration objective. Metamorphic reasoning evaluation
   itself is established, so exact-transform evaluation alone is not claimed
   as a new general methodology here.

10. **RoleQ: Role-Aligned Scales for Converting Language Models to Ternary**,
    Knittel, Helbling and Pfister, arXiv:2610.21808v1 (15 October 2026).
    https://arxiv.org/html/2610.21808
    Official record and section 3.4 read. Its scale distillation uses positive
    log scales, frozen codes, cached teacher states and next-token KL; it also
    trains norm gains. These implementation ingredients are established and
    must not be presented as our method's novelty. Our controlled comparison
    instead isolates canonical semantic-pair regularization and transfer.

11. **The Illusion of Equivalency: Statistical Characterization of Quantization
    Effects in LLMs**, Rababah et al., arXiv:2607.08734v2 (21 August 2026).
    https://arxiv.org/html/2607.08734v2
    Added during the final adversarial search before public release. Read the
    abstract, introduction, sections III-D and IV, and experimental setup.
    Correctness Agreement measures the intersection of correct decisions of
    the original and quantized model on the same input. Thus neither the
    observation that equal accuracy hides changed decisions nor measuring
    joint correctness is new. Here the two members of a pair are distinct,
    explicitly mapped presentations scored by the same model. The retrieved
    method does not fit a semantic-pair calibration loss or evaluate transfer
    to held-out transformation compositions.

12. **Unsupervised Data Augmentation for Consistency Training**, Xie et al.,
    NeurIPS 2020, https://arxiv.org/abs/1904.12848 . The original abstract
    explicitly covers invariant predictions under augmented inputs in language
    and vision. **AugMix: A Simple Data Processing Method to Improve Robustness
    and Uncertainty**, Hendrycks et al., arXiv:1912.02781v2,
    https://arxiv.org/html/1912.02781v2 , uses JS consistency between original
    and augmented predictions (section 3 and algorithm read). Thus neither
    invariance training nor the JS regularizer is novel. The contribution is
    assessed as a controlled application/evaluation in quantized LLMs, with
    exact answer transport and matched inputs, rather than a new loss family.

## Searches and candidate claim

Web searches covered quantization + semantic equivariance, paired consistency,
metamorphic tests, paraphrase/calibration objectives and prompt robustness.
The initial topic-selection search used alphaXiv and Exa for
direct counterexamples. Later passes checked newer fidelity/uncertainty and
explanation-preserving objectives to avoid an overly broad novelty claim.
An additional Exa search explicitly sought semantic-pair JS calibration with
frozen quantizer scales, recovering RoleQ and the published EAQuant record.
Both were checked against their original pages; neither retrieved method
describes our exact semantic-input-pair calibration/evaluation combination.

Candidate contribution: an oracle-verified, split-controlled assessment of
quantization-induced changes in canonical answer correspondence, followed
conditionally by a same-data/same-update paired-calibration comparison on
held-out transformations. Exact agreement alone is not correctness; unchanged
average accuracy alone is not pointwise fidelity. New pair inconsistency
necessarily entails at least one changed pointwise prediction, so it is not
an independent failure mechanism proved merely by a consistency score.

No direct same-objective/same-evaluation match was found in the retrieved
sources. This supports investigating the question, not claiming priority or
superiority. A final targeted Exa search and the original Rababah et al.
paper were checked before public manuscript finalization; this added a
further relevant behavioral-evaluation citation without establishing a
direct objective/evaluation match.

The additional KernelQuant result narrows the claim further: combine exact
semantic correspondence with a matched calibration/evaluation contract, rather
than claiming either relation preservation or metamorphic testing in isolation.
