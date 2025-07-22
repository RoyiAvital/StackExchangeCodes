# Robust Ellipse Fit

The Algebraic Representation of an Ellipse for a point $\left( x, y \right)$ on the ellipse is given by:

$$ a {x}^{2} + b x y + c {y}^{2} + d x + e y + f = 0 \Leftrightarrow \boldsymbol{x}^{\top} \boldsymbol{A} + \boldsymbol{b}^{\top} \boldsymbol{x} + f $$

For a set of points $\left\{ {\left[ {x}_{i}, {y}_{i} \right]}^{T} \right\}_{i = 1}^{n}$ one could build the Scatter Matrix:

$$ \boldsymbol{D} = \begin{bmatrix} \text{---} & \boldsymbol{d}_{1} & \text{---} \\ \text{---} & \boldsymbol{d}_{2} & \text{---} \\ \text{---} & \vdots & \text{---} \\ \text{---} & \boldsymbol{d}_{n} & \text{---} \end{bmatrix}, \;  \boldsymbol{d}_{i} = {\left[ {x}_{i}^{2}, {x}_{i} {y}_{i}, {y}_{i}^{2}, {x}_{i}, {y}_{1}, 1 \right]} $$

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

Where 
 - $\boldsymbol{p} = {\left[ a, b, c, d, e, f \right]}^{\top}$ - The vector of parameters.
 - $\boldsymbol{A} = \begin{bmatrix} {q}_{1} & \frac{{q}_{2}}{2} \\ \frac{{q}_{2}}{2} & {q}_{3} \end{bmatrix}$.
 - The function $f : \mathbb{R}^{6} \to \mathbb{R}$ is a loss function which promotes lower values. Conceptually behaves like $\left\| \boldsymbol{D} \boldsymbol{p} \right\|$.
 - The constraint $\boldsymbol{A} \in \mathbb{S}_{+}^{2}$ means the matrix is SPSD (Symmetric Positive Semi Definite) which forces the solution to be an ellipse or parabola (See [Matrix Representation of Conic Sections][1]).
 - The constraint $\operatorname{tr} \left( \boldsymbol{A} \right) = 1$ solve the scaling issue and guarantees an ellipse as it forces the sum of eigenvalues to be 1.

 Define:
 
  * $\mathcal{A} \left( \boldsymbol{p} \right) = \boldsymbol{A}$ - A linear operator which extracts the matrix $\boldsymbol{A}$ from a vector.
  * $\mathcal{C} = \left\{ \boldsymbol{A} \mid \boldsymbol{A} \in \mathbb{S}_{+}^{2}, \operatorname{tr} \left( \boldsymbol{A} \right) = 1 \right\}$ - The set of matrices which obey the constraints.
  * $\mathcal{A}^{-1} \left( \boldsymbol{A}, \boldsymbol{p} \right)$ A linear operator which updates the 3 elements of the vector $\boldsymbol{p}$ such that ${p}_{1} = {A}_{1, 1}, {p}_{2} = {A}_{1, 2}, {p}_{3} = {A}_{2, 2}$.
  * $\boldsymbol{K} = \boldsymbol{D}^{\top} \boldsymbol{D}$ (Can be pre calculated).

Then the problem can be rewritten:

$$
\begin{align*}
\arg \min_{\boldsymbol{p}} \quad & f \left( \boldsymbol{D} \boldsymbol{p} \right) \\
\text{subject to} \quad & \begin{aligned} 
\mathcal{A} \left( \boldsymbol{p} \right) & \in \mathcal{C} \\
\end{aligned}
\end{align*}
$$

The concept is to search methods which can handle large condition number of $\boldsymbol{D}$.  
It can be done mostly by avoiding steps which require its inverse directly.


## Least Squares Based Optimization 

The _Least Squares_ variant sets $f$ to be the Least Squares loss function:

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
Since $\operatorname{tr} \left( \boldsymbol{U}^{T} \boldsymbol{A} \boldsymbol{U} \right) = \operatorname{tr} \left( \boldsymbol{A} \boldsymbol{U} \boldsymbol{U}^{T} \right) = \operatorname{tr} \left( \boldsymbol{A} \right)$ the projection is given by:

$$ \operatorname{P}_{\mathcal{C}} \left( \boldsymbol{B} \right) = \boldsymbol{U} \operatorname{Diag} \left( \hat{\boldsymbol{b}} \right) \boldsymbol{U}^{T} $$

Where
 - $\boldsymbol{B} = \boldsymbol{U} \operatorname{Diag} \left( \boldsymbol{b} \right) \boldsymbol{U}^{T}$ - The [Eigen Decomposition][2] of $\boldsymbol{B}$.
 - $\hat{\boldsymbol{b}}$ - The projection of $\boldsymbol{b}$ onto the Unit Simplex.

In order to apply the solution on the vector, the result of the projection should be inserted into the projected vector.

### Solution by (Accelerated) Projected Gradient Descent

With $f = \frac{1}{2} {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{2}^{2}$ the _Projected Gradient Descent_:

 - Set $\boldsymbol{p}_{0}$.
 - For $k = 1, 2, \ldots, K$:
    1. $\boldsymbol{g}_{k} = {\nabla}_{\boldsymbol{p}} f \left( \boldsymbol{p}_{k - 1} \right)$.
    1. $\boldsymbol{p}_{k} = \boldsymbol{p}_{k - 1} - \mu \boldsymbol{g}_{k}$.
    2. $\boldsymbol{p}_{k} = \mathcal{A}^{-1} \left( \operatorname{P}_{\mathcal{C}} \left( \mathcal{A} \left( \boldsymbol{p}_{k} \right) \right), \boldsymbol{p}_{k} \right)$.
 - Return $\boldsymbol{p}_{K}$.

Remarks:
 - ${\nabla}_{\boldsymbol{p}} f \left( \boldsymbol{p} \right) = \boldsymbol{K} \boldsymbol{p}$.
 - One may use accelerated version of the algorithm.
 - One may check for convergence conditions on each iteration.

### Solution by Separable Prox Method (Douglas Rachford Splitting)

The _Douglas Rachford Splitting_ (See [Proximal Splitting Methods in Signal Processing][3], at the _Douglas Rachford Splitting_ chapter) optimization model:

$$ \arg \min_{\boldsymbol{p}} f \left( \boldsymbol{p} \right) + g \left( \boldsymbol{p} \right) $$

The solver iterations:

 - Set $\boldsymbol{q}_{0}, \gamma > 0, 0 < \lambda < 1$.
 - For $k = 1, 2, \ldots, K$:
    1. $\boldsymbol{p}_{k} = \operatorname{prox}_{\gamma f} \left( \boldsymbol{q}_{k - 1} \right)$.
    2. $\boldsymbol{q}_{k} = \boldsymbol{q}_{k - 1} + \lambda \left( \operatorname{prox}_{\gamma g} \left( 2 \boldsymbol{p}_{k} - \boldsymbol{q}_{k - 1} \right) - \boldsymbol{p}_{k} \right)$.
 - Return $\operatorname{prox}_{\gamma f} \left( \boldsymbol{p}_{K} \right)$.

Remarks:
 - One may use ${\lambda}_{k}$ to adapt its value per iteration.
 - One may check for convergence conditions on each iteration.

Setting:

 - $f \left( \boldsymbol{p} \right) = \frac{1}{2} {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{2}^{2}$ - The objective function.
 - $g \left( \boldsymbol{p} \right) = {I}_{\mathcal{C}} \left( \boldsymbol{p} \right)$ - The indicator function over $\mathcal{C}$.

The Proximal Operator of each function is given by:

$$ \operatorname{prox}_{\gamma f} \left( \boldsymbol{y} \right) = {\left( \gamma \boldsymbol{K} + \boldsymbol{I} \right)}^{-1} \boldsymbol{y}, \; \operatorname{prox}_{\gamma g} \left( \boldsymbol{y} \right) = \mathcal{A}^{-1} \left( \operatorname{P}_{\mathcal{C}} \left( \mathcal{A} \left( \boldsymbol{y} \right) \right), \boldsymbol{y} \right) $$

Remarks:
 * The matrix $\gamma \boldsymbol{K} + \boldsymbol{I}$ is SPD hence its Cholesky Decomposition can be pre calculated to accelerate the system of equations solution.


## Least Absolute Deviation Based Optimization 

The _Least Absolute Deviation_ variant sets $f$ to be the ${L}_{1}$ based loss function:

$$
\begin{align*}
\arg \min_{\boldsymbol{p}} \quad & {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{1} \\
\text{subject to} \quad & \begin{aligned} 
\mathcal{A} \left( \boldsymbol{p} \right) & \in \mathcal{C} \\
\end{aligned}
\end{align*}
$$

The motivation for this loss is its reduced sensitivity to _outliers_ compared to the ${L}_{2}$ based loss functions.

### Solution by (Accelerated) Projected Sub Gradient Descent

With $f = \frac{1}{2} {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{1}$ the _Projected Sub Gradient Descent_:

 - Set $\boldsymbol{p}_{0}$.
 - For $k = 1, 2, \ldots, K$:
    1. $\boldsymbol{g}_{k} = {\partial}_{\boldsymbol{p}} f \left( \boldsymbol{p}_{k - 1} \right)$.
    1. $\boldsymbol{p}_{k} = \boldsymbol{p}_{k - 1} - \mu \boldsymbol{g}_{k}$.
    2. $\boldsymbol{p}_{k} = \mathcal{A}^{-1} \left( \operatorname{P}_{\mathcal{C}} \left( \mathcal{A} \left( \boldsymbol{p}_{k} \right) \right), \boldsymbol{p}_{k} \right)$.
 - Return $\boldsymbol{p}_{K}$.

Remarks:
 - ${\partial}_{\boldsymbol{p}} f \left( \boldsymbol{p} \right) = \boldsymbol{D}^{\top} \operatorname{sign} \left( \boldsymbol{D} \boldsymbol{p} \right)$ - A _Sub Gradient_ of the objective function.
 - One may use accelerated version of the algorithm.
 - One may check for convergence conditions on each iteration.
 - Convergence is relatively slow.


### Solution by Primal Dual Hybrid Gradient (Chambolle Pock)

The _Chambolle Pock_ optimization model:

$$ \arg \min_{\boldsymbol{p}} f \left( \boldsymbol{D} \boldsymbol{p} \right) + g \left( \boldsymbol{p} \right) $$

The solver iterations:

 - Set $\boldsymbol{p}_{0}, \boldsymbol{q}_{0}, \bar{\boldsymbol{p}}_{0} = \boldsymbol{p}_{0}, \\theta \in \left[ 0, 1 \right], \tau > 0, \sigma > 0$.
 - For $k = 1, 2, \ldots, K$:
    1. $\boldsymbol{q}_{k} = \operatorname{prox}_{\sigma {f}^{\ast}} \left( \boldsymbol{y}_{k - 1} + \sigma \boldsymbol{D} \bar{\boldsymbol{p}}_{k - 1} \right)$.
    2. $\boldsymbol{p}_{k} = \operatorname{prox}_{\tau g} \left( \boldsymbol{p}_{k - 1} - \tau \boldsymbol{D}^{\top} \boldsymbol{q}_{k} \right)$.
    3. $\bar{\boldsymbol{p}}_{k} = \boldsymbol{p}_{k} + \theta \left( \boldsymbol{p}_{k} - \boldsymbol{p}_{k - 1} \right)$.
 - Return $\boldsymbol{p}_{K}$.

Remarks:
 * The algorithm requires the $\operatorname{prox}$ of the conjugate function of $f$.
 * It converges for $\tau \sigma {\left\| \boldsymbol{D} \right\|}_{2}^{2} \leq 1$ with ${\left\| \cdot \right\|}_{2}^{2}$ being the squared _Spectral Norm_ of the operator (Square of its largest singular value).
 * There are some methods to accelerate it with acceleration methods and / or adaptive step methods (See [An Introduction to Continuous Optimization for Imaging][4], [Adaptive Primal Dual Splitting Methods for Statistical Learning and Image Processing][5]).
 * One may check for convergence conditions on each iteration.

Setting:

 - $f \left( \boldsymbol{p} \right) = {\left\| \boldsymbol{D} \boldsymbol{p} \right\|}_{1}$ - The objective function.
 - $g \left( \boldsymbol{p} \right) = {I}_{\mathcal{C}} \left( \boldsymbol{p} \right)$ - The indicator function over $\mathcal{C}$.

The Proximal Operator of each function is given by:

$$ \operatorname{prox}_{\gamma {f}^{\ast}} \left( \boldsymbol{y} \right) = \operatorname{P}_{ \mathcal{B}_{{\left\| \cdot \right\|}_{\infty}} }  \left( \boldsymbol{y}\right), \; \operatorname{prox}_{\gamma g} \left( \boldsymbol{y} \right) = \mathcal{A}^{-1} \left( \operatorname{P}_{\mathcal{C}} \left( \mathcal{A} \left( \boldsymbol{y} \right) \right), \boldsymbol{y} \right) $$

Remarks:
 * The conjugate of the ${L}_{1}$ Norm: ${f}^{\ast} \left( \boldsymbol{y} \right) = \sup \boldsymbol{y}^{\top} \boldsymbol{x} - {\left\| \boldsymbol{x} \right\|}_{1}$. If ${\left\| \boldsymbol{x} \right\|}_{\infty} \leq 1$, then there exists an $\boldsymbol{x}$ such that $\boldsymbol{y}^{\top} \boldsymbol{x} = {\left\| \boldsymbol{x} \right\|}_{1}$. In this case, the supremum is $0$. If ${\left\| \boldsymbol{y} \right\|}_{\infty} > 1$ then there exist an $\boldsymbol{x}$ that makes $\boldsymbol{y}^{\top} \boldsymbol{x} - {\left\| \boldsymbol{x} \right\|}_{1}$ arbitrarily large. In this case, the supremum is infinity. This is the definition of the Indicator function over $\mathcal{B}_{{\left\| \cdot \right\|}_{\infty}}$. Hence its Prox operator is the projection onto the ball set.
 * The motivation for the algorithm is avoiding solving the Prox problem over the ${L}_{1}$ Norm composed with linear operator which does not have a closed form solution. This is the reason to prefer this algorithm over ADMM for this case as its sub problems are easier to solve.


  [1]: https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
  [2]: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
  [3]: https://arxiv.org/abs/0912.3522
  [4]: https://hal.science/hal-01346507
  [5]: https://papers.nips.cc/paper_files/paper/2015/hash/cd758e8f59dfdf06a852adad277986ca-Abstract.html