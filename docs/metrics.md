# Canonical correspondence and its limits

Let a displayed answer index be j in {0,1,2,3}. Each transformed record stores
a bijection m from displayed indices to the original option identities. A
prediction a is compared as m(a), never as its displayed A/B/C/D label.
For a probability vector p, its canonical vector has p_c[m(j)] = p[j].
This convention must be tested with a non-self-inverse permutation.

For base problem i and transformation t, define correctness Y_it and canonical
prediction A_it. Correspondence C_it = 1[A_i0 = A_it] is distinct from joint
correctness J_it = Y_i0 Y_it. With exactly one correct canonical identity,
joint correctness implies correspondence; the converse need not hold.
Consistent-but-wrong pairs satisfy C_it=1 and Y_i0=Y_it=0.

For full-precision and quantized predictions, let F_i0 and F_it indicate an
individual canonical answer change. If neither changes, correspondence cannot
change. Consequently the absolute change in aggregate correspondence is at
most the mean of F_i0+F_it (a union-bound diagnostic). This is not a new theorem
or proof of a separate reasoning mechanism. Equal aggregate accuracy can hide
different item-level errors and different correspondence.

Summarize separately by model, precision, split, task and transformation.
For pooled comparisons, average within each base problem before resampling
base problems; never treat multiple variants of one problem as independent.
Report unavailable transformations explicitly. Calibration seeds are distinct
fits, not additional independent test examples.

Any future paired-calibration objective must distinguish imposing additional
invariance from preserving the original model: the teacher may itself disagree
across transformations. A teacher-matching term and a paired-invariance term
can conflict. Measure teacher fidelity, correspondence and ground-truth
accuracy together, including cases where extra agreement is jointly wrong.
An improvement in agreement at the cost of accurate answers is not reported
as a reasoning improvement.
