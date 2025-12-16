# PPGR-GSR

\subsection{Complexity analysis}
At the pre-processing stage, the complexity of searching similar patches is \( O(NM^2b) \).
For dictionary learning, the time complexity for performing PCA on each group \( \mathbf{Y}_{G_i} \in \mathbb{R}^{p \times k} \) is \( O(b^2 m + m b^2) \). 
The sparse coding optimization, solved via the ADMM, involves solving the Sylvester equation (\eqref{eq13}), which requires \( O((b m)^3) \) operations. 
Therefore, the overall time complexity of the proposed method is \( O(NM^2b + N \cdot (b^2 m + m b^2) + k \cdot (b m)^3) \).
