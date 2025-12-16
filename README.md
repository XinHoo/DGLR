# PPGR-GSR

\subsection{Complexity analysis}
Let $n_g$ denote the number of groups (reference patches). For each group, we have
$\mathbf{Y}_i \in \mathbb{R}^{b \times m}$, where $b$ is the patch dimension and $m$ is the number of matched similar patches.
Let $T_{\mathrm{out}}$ be the number of outer BCD iterations and $T$ be the maximum number of inner ADMM iterations per group.

\textbf{Block matching (BM).}
Searching $m$ similar patches for each reference patch within an $M \times M$ search window has a complexity of
$O(n_g \, M^2 \, b)$ per outer iteration, where $M^2$ is the number of candidates and $b$ is the cost of computing patch distances.
Thus, over $T_{\mathrm{out}}$ outer iterations, BM costs $O(T_{\mathrm{out}}\, n_g \, M^2 \, b)$.

\textbf{Graph construction.}
Constructing the patch graph over $m$ patches costs $O(m^2 b)$ per group for a fully connected graph (pairwise distances in $\mathbb{R}^b$).
Constructing the pixel-position graph over $b$ intra-patch locations costs $O(b^2 m)$ per group for a fully connected graph
(pairwise distances in $\mathbb{R}^m$).
Therefore, the total graph construction cost per outer iteration is
$O\!\big(n_g (m^2 b + b^2 m)\big)$, and over $T_{\mathrm{out}}$ iterations it becomes
$O\!\big(T_{\mathrm{out}}\, n_g (m^2 b + b^2 m)\big)$.
(If $k$-NN graphs are employed with approximate neighbor search, the empirical cost can be reduced, but the worst-case
pairwise-distance complexity remains dominated by the above terms.)

\textbf{Dictionary update.}
Computing the truncated SVD of $\mathbf{Y}_i \in \mathbb{R}^{b \times m}$ to form
$\mathbf{H}_i=\mathbf{U}_i(:,1\!:\!r)$ costs $O\!\big(b m \min(b,m)\big)$ per group in the worst case
(e.g., $O(bm^2)$ when $b \ge m$).
Thus, over all groups and outer iterations, the dictionary update costs
$O\!\big(T_{\mathrm{out}}\, n_g \, b m \min(b,m)\big)$.

\textbf{Sparse coding (ADMM).}
The dominant cost per ADMM iteration is updating $\mathbf{S}_i \in \mathbb{R}^{r \times m}$ by solving
the linear system in \eqref{eq14}--\eqref{eq15}:
\(
\mathbf{K}\operatorname{vec}(\mathbf{S}_i^{k+1})=\operatorname{vec}(\mathbf{C}),
\)
where $\mathbf{K}\in\mathbb{R}^{(rm)\times(rm)}$ is symmetric positive definite and is solved via a Cholesky factorization.
Since $\mathbf{K}$ remains fixed within the inner ADMM loop for each group, its Cholesky factorization is computed once per group,
with complexity $O\!\big((rm)^3\big)$ (and memory $O\!\big((rm)^2\big)$).
Each ADMM iteration then requires two triangular solves, costing $O\!\big((rm)^2\big)$, while the $\mathbf{Z}$-update (soft-thresholding)
and dual update are elementwise and cost $O(rm)$.
Therefore, the per-group ADMM complexity is
\[
O\!\Big((rm)^3 + T\big((rm)^2 + rm\big)\Big)\approx O\!\Big((rm)^3 + T(rm)^2\Big).
\]
Over all groups and outer iterations, the total sparse coding complexity is
\[
O\!\Big(T_{\mathrm{out}}\, n_g \big((rm)^3 + T(rm)^2\big)\Big).
\]

\textbf{Overall complexity.}
Combining the above terms, the overall computational complexity of PPGR-GSR is
\[
O\!\Big(
T_{\mathrm{out}}\, n_g \big(
M^2 b
+ m^2 b + b^2 m
+ b m \min(b,m)
+ (rm)^3 + T(rm)^2
\big)
\Big),
\]
where $b$ (patch dimension) and the group sizes $(m,r)$ are typically much smaller than the image size, making the per-group computations tractable.

“We solve the Sylvester equation in (13) using a standard Schur/Bartels–Stewart solver; since (\mathbf A) and (\mathbf B) are fixed within the inner ADMM loop, their Schur decompositions are computed once per group, and each ADMM iteration costs (O(r^2m+rm^2)).”

