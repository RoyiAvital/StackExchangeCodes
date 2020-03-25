# Matrix Form of Image Gradient

Given an image $ X \in \mathbb{R}^{m \times n} $ its 2D Gradient Vector $ \boldsymbol{d} $ is given by:

$$ \boldsymbol{d} = \begin{bmatrix} \boldsymbol{d}^{v} \\ \boldsymbol{d}^{h} \end{bmatrix}, \quad {d}^{v}_{k} = {X}_{i, j} - {X}_{i + 1, j}, \quad {d}^{h}_{l} = {X}_{i, j} - {X}_{i, j + 1} $$

Where $ k = i + \left( j - 1 \right) \left( m - 1 \right), \; i \in \left\{ 1, 2, \ldots, m - 1 \right\}, j \in \left\{ 1, 2, \ldots, n - 1 \right\} $ and $ l = i + \left( j - 1 \right) \left( m \right), \; i \in \left\{ 1, 2, \ldots, m \right\}, j \in \left\{ 1, 2, \ldots, n - 1 \right\} $.  
In MATLAB Notatiotn:

$$ \boldsymbol{d}^{v} = \operatorname{vec} \left( {X}_{1:m - 1, :} - {X}_{2:m, :} \right), \quad \boldsymbol{d}^{h} = \operatorname{vec} \left( {X}_{:, 1:n - 1} - {X}_{:, 2:n} \right) $$

Where $ \operatorname{vec} \left( \cdot \right) $ is the [Column Wise Vectorization Operator][01].

## Building the Matrix Form

We are after $ D $ such that $ \boldsymbol{d} = D \operatorname{vec} \left( X \right) = D \boldsymbol{x} $.  
The easiest way is to think about the the operator working on the vertical derivative:

$$ {X}^{v} = {X}_{1:m - 1, :} - {X}_{2:m, :} = {D}^{v} X $$

It is clear that the matrix is given by:

$$ {D}^{v} = \begin{bmatrix}
1 & -1 &  &  & \\ 
 & 1 & -1 &  & \\ 
 &  & \ddots  & \ddots & \\ 
 &  &  & 1 & -1
\end{bmatrix} \in \mathbb{R}^{ \left( m - 1 \right) \times m} $$

Doing the same for the horizontal derivative:

$$ {X}^{h} = {X}_{:, 1:n - 1} - {X}_{:, 2:n} = X {D}^{h} \Rightarrow {D}^{h} = \begin{bmatrix}
1 &  &  &  \\ 
-1 & 1 &  &  \\ 
 & -1 & \ddots  & \\ 
 &  & \ddots & 1 \\
 &  &  & -1
\end{bmatrix} \in \mathbb{R}^{n \times \left( n - 1 \right)} $$

Now, Let's assume we have a generator for the matrix $ {D}^{v} $ for a given $ m $. We'll denote it by $ T \left( m \right) $ it is clear that in order to generate $ {D}^{h} $ for some $ n $ all needed it to transpose the output of $ T \left( \cdot \right) $ for the same $ n $. So $ {T \left( n \right)}^{T} = {D}^{h} \in \mathbb{R}^{n \times \left( n - 1 \right)} $.

So now we know to generate both matrices with the same generator. From here to get $ D $ one should use the following property of the [Kronecker Product][02]:

$$ A X B = C \Leftrightarrow \operatorname{vec} \left( C \right) = \operatorname{vec} \left( A X B \right) = \left( {B}^{T} \otimes A \right) \operatorname{vec} \left( X \right) $$

Applying the above to the vertical equation:

$$ {X}^{v} = {D}^{v} X {I}_{n} \Rightarrow \boldsymbol{d}^{v} = \left( {I}_{n}^{T} \otimes {D}^{v} \right) \boldsymbol{x} = \left( {I}_{n} \otimes T \left( m \right) \right) \boldsymbol{x} $$

Doing the same for the vertical case: 

$$ {X}^{h} = {I}_{m} X {D}^{h} \Rightarrow \boldsymbol{d}^{v} = \left( {{D}^{h}}^{T} \otimes {I}_{m} \right) \boldsymbol{x} = \left( {T \left( n \right)^{T}}^{T} \otimes {I}_{m} \right) \boldsymbol{x} = \left( T \left( n \right) \otimes {I}_{m} \right) \boldsymbol{x} $$

So in summary:

$$ D = \begin{bmatrix} {I}_{n} \otimes T \left( m \right) \\ T \left( n \right) \otimes {I}_{m} \end{bmatrix} \Rightarrow \boldsymbol{d} = D \boldsymbol{x} $$

This is implemented in `CreateGradientOperator()`.



  [01]: https://en.wikipedia.org/wiki/Vectorization_(mathematics)
  [02]: https://en.wikipedia.org/wiki/Kronecker_product