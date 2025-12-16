# PPGR-GSR

\subsection{Complexity analysis}
Let $n_g$ denote the number of groups (reference patches). For each group, we have
$\mathbf{Y}_i \in \mathbb{R}^{b \times m}$, where $b$ is the patch dimension and $m$ is the number of matched similar patches.

\textbf{Block matching.}
Searching $m$ similar patches within a $M \times M$ search window has a complexity of
$O(n_g \, M^2 \, b)$, where $M^2$ is the number of candidates and $b$ is the cost of computing patch distances.

\textbf{Graph construction.}
Constructing the patch graph over $m$ patches costs $O(m^2 b)$ if fully connected (pairwise distances in $\mathbb{R}^b$).
Constructing the pixel-position graph over $b$ intra-patch locations costs $O(b^2 m)$ if fully connected (pairwise distances in $\mathbb{R}^m$).
If $k$-NN graphs are used, these costs reduce to $O(m k_g b)$ and $O(b k_p m)$, respectively.

\textbf{Dictionary update.}
Computing the truncated SVD of $\mathbf{Y}_i \in \mathbb{R}^{b \times m}$ to form $\mathbf{H}_i=\mathbf{U}_i(:,1\!:\!r)$ has a complexity
$O\!\big(b m \min(b,m)\big)$ per group (or $O(b m^2)$ when $b \ge m$).

\textbf{Sparse coding (ADMM).}
The dominant step in each ADMM iteration is updating $\mathbf{S}_i \in \mathbb{R}^{r \times m}$ by solving the linear system in \eqref{eq14}--\eqref{eq15}.
The coefficient matrix $\mathbf{K} \in \mathbb{R}^{(rm)\times(rm)}$ is fixed for a given group (with fixed $\mu$, $\mathbf{L}$, and $\mathbf{L}_{\mathrm{pix}}$),
thus its Cholesky factorization is computed once with cost $O((rm)^3)$, and each subsequent iteration requires two triangular solves with cost $O((rm)^2)$.
The $\mathbf{Z}_i$ and dual updates are elementwise and cost $O(rm)$ per iteration.
Therefore, the per-group ADMM complexity is $O\big((rm)^3 + T (rm)^2\big)$ (dominant terms).

Overall, the total complexity over all groups is the sum of the above costs multiplied by $n_g$.

