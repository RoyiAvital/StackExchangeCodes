# Robust Ellipse Fit

A _Convex_ formulation for Ellipse Fitting model:

$$
\begin{align*}
\arg \min_{\boldsymbol{p}} \quad & f \left( \boldsymbol{D} \boldsymbol{p} \right) \\
\text{subject to} \quad & \begin{aligned} 
\boldsymbol{A} & \in \mathbb{S}_{+}^{2} \\
\operatorname{tr} \left( \boldsymbol{A} \right) & = 1 \\
\end{aligned}
\end{align*}
$$

Where $\boldsymbol{A} = \begin{bmatrix} {q}_{1} & \frac{{q}_{2}}{2} \\ \frac{{q}_{2}}{2} & {q}_{3} \end{bmatrix}$.

 - The function $f : \mathbb{R}^{6} \to \mathbb{R}$ is a loss function which promotes lower values. Conceptually behaves like $\left\| \boldsymbol{D} \boldsymbol{p} \right\|$.
 - The constraint $\boldsymbol{A} \in \mathbb{S}_{+}^{2}$ means the matrix is SPSD (Symmetric Positive Semi Definite) which forces the solution to be an ellipse or parabola (See [Matrix Representation of Conic Sections][1]).
 - The constraint $\operatorname{tr} \left( \boldsymbol{A} \right) = 1$ solve the scaling issue and guarantees an ellipse as it forces the sum of eigenvalues to be 1.

 Define:
 
  * $\mathcal{A} \left( \boldsymbol{p} \right) = \boldsymbol{A}$ - A linear operator which extracts the matrix $\boldsymbol{A}$ from a vector.
  * $\mathcal{C} = \left\{ \boldsymbol{A} \mid \boldsymbol{A} \in \mathbb{S}_{+}^{2}, \operatorname{tr} \left( \boldsymbol{A} \right) = 1 \right\}$ The set of matrices which obey the constraints.

Then the problem can be rewritten:

$$
\begin{align*}
\arg \min_{\boldsymbol{p}} \quad & f \left( \boldsymbol{D} \boldsymbol{p} \right) \\
\text{subject to} \quad & \begin{aligned} 
\mathcal{A} \left( \boldsymbol{p} \right) & \in \mathcal{C} \\
\end{aligned}
\end{align*}
$$


## Least Squares Based Optimization 

The _Least Squares variant sets $f$ to be the Least Squares loss function:

$$
\begin{align*}
\arg \min_{\boldsymbol{p}} \quad & \frac{1}{2} {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{2}^{2} \\
\text{subject to} \quad & \begin{aligned} 
\mathcal{A} \left( \boldsymbol{p} \right) & \in \mathcal{C} \\
\end{aligned}
\end{align*}
$$

Let $\operatorname{P}_{\mathcal{C}} : \mathbb{R}^{2} \to \mathbb{R}^{2}$ be the projection operator of a symmetric matrix onto $\mathcal{C}$.  

Then the problem above can be solved in various methods.

### Projection onto the Set $\mathcal{C}$

Given a symmetric matrix $\boldsymbol{B}$ (Built form the 3 elements of the vector in our case):

$$
\begin{align*}
\arg \min_{\boldsymbol{A} \in \mathcal{C}} \frac{1}{2} {\left\| \boldsymbol{A} - \boldsymbol{B} \right\|}_{F}^{2} & = \arg \min_{\boldsymbol{A} \in \mathcal{C}} \frac{1}{2} {\left\| \boldsymbol{U}^{T} \boldsymbol{A} \boldsymbol{U} - \operatorname{Diag} \left( \boldsymbol{b} \right) \right\|}_{F}^{2} && \text{Eigen decomposition of $\boldsymbol{B}$} \\
& = \arg \min_{\boldsymbol{A} \in \mathcal{C}} \frac{1}{2} {\left\| \boldsymbol{C} - \operatorname{Diag} \left( \boldsymbol{b} \right) \right\|}_{F}^{2} && \text{} \\
& = \arg \min_{\boldsymbol{A} \in \mathcal{C}} \sum_{i} {\left( {C}_{i, i} - \boldsymbol{b}_{i} \right)}^{2} + 2 \sum_{i < j} {C}_{i, j}^{2}
\end{align*}
$$

In order to minimize the sum, the off diagonal element of $\boldsymbol{C}$ must be zero. In order to do so, the matrix $\boldsymbol{U}$ must also diagonalize the matrix $\boldsymbol{A}$.  
Since $\operatorname{tr} \left( \boldsymbol{U}^{T} \boldsymbol{A} \boldsymbol{U} \right) = \operatorname{tr} \left( \boldsymbol{A} \boldsymbol{U} \boldsymbol{U}^{T} \right) = \operatorname{tr} \left( \boldsymbol{A} \right)$ we have that the solution is given by $\boldsymbol{U} \operatorname{Diag} \left( \hat{\boldsymbol{b}} \right) \boldsymbol{U}^{T}$ where $\hat{\boldsymbol{b}}$ is the projection of $\boldsymbol{b}$ onto the Unit Simplex.

In order to apply the solution on the vector, the result of the projection should be inserted into the projected vector.

### Solution by (Accelerated) Projected Gradient Descent

Defining $\mathcal{A}^{-1} \left( \boldsymbol{A}, \boldsymbol{p} \right)$ as the operation of updating the 3 elements of the vector $\boldsymbol{p}$ such that ${p}_{1} = {A}_{1, 1}, {p}_{2} = {A}_{1, 2}, {p}_{3} = {A}_{2, 2}$.

Then the _Projected Gradient Descent_ can be defined:

 - Set $\boldsymbol{p}_{0}$.
 - For $k = 1, 2, \ldots, K$:
    1. $\boldsymbol{p}_{k + 1} = \boldsymbol{p}_{k} - \mu {\nabla}_{\boldsymbol{p}} \frac{1}{2} {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{2}^{2}$.
    2. $\boldsymbol{p}_{k + 2} = \mathcal{A}^{-1} \left( \operatorname{P}_{\mathcal{C}} \left( \mathcal{A} \left( \boldsymbol{p}_{k + 1} \right) \right), \boldsymbol{p}_{k + 1} \right)$.
 - Return $\boldsymbol{p}_{K}$.

Remarks:
 - One may use accelerated version of the algorithm.
 - One may check for convergence conditions on each iteration.

### Solution by Separable Prox Method






  [1]: https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections