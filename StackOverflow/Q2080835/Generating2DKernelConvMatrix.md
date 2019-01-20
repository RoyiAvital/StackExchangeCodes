# Generating 2D Kernel Convolution Matrix

In this document the generation of a Convolution Matrix for a 2D Convolution Kernel will be described.  
The idea is building a convolution matrix which matches the form of 3 convolutions shapes:

 *  Full.
 *  Same.
 *  Valid.

## 1D Convolution Matrix

## 2D Convolution Matrix

### Definitions

 *  2D Signal $ f \left[ m, n \right] $ with finite support $ {\Omega}_{f} = \left\{ \left[ m, n \right] \mid 1 \leq m \leq M, 1 \leq n \leq N \right\} $.
 *  Convolution Kernel $ h \left[ m, n \right] $ with finite support $ {\Omega}_{h} = \left\{ \left[ m, n \right] \mid 1 \leq m \leq P, 1 \leq n \leq Q \right\} $.

 ### 2D Convolution
 The [2D convolution](https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution) (Full Convolution) between $ f \left[ m, n \right] $ and $ h \left[ m, n \right] $ is given by:

$$\begin{align*} g \left[ m ,n \right] = \left( f \ast h \right) \left[ m, n \right] & = \sum_{k}^{M} \sum_{r}^{N} f \left[ k , r \right] h \left[ m - k + 1, n - r + 1 \right] & \text{} \\ 
& = \sum_{k}^{P} \sum_{r}^{Q} h \left[ k , r \right] f \left[ m - k + 1, n - r + 1 \right] & \text{} \\
& = \left( h \ast f \right) \left[ m, n \right] & \text{Convolution is commutative}
\end{align*}$$

 With the support of $ g \left[ m, n \right] $ given by $ {\Omega}_{g} = \left\{ \left[ m, n \right] \mid 1 \leq m \leq M + P - 1, 1 \leq n \leq N + Q - 1 \right\} $.

 Since the operation is linear using the column stack representation of $ f \left[ m, n \right] $ (Basically the [Vecotrization](https://en.wikipedia.org/wiki/Vectorization_(mathematics)) operator) one could write:

 $$ g = H f $$

 Where $ g $ is the Column Stack representation of $ g \left[ m, n \right] $, $ f $ is the column stack representation of $ f \left[ m, n \right] $ and $ H \in \mathbb{R}^{\left( M + P - 1 \right) \times \left( N + Q - 1 \right)} $ is the 2D Convolution Matrix.

 In order to understand how to build the matrix $ H $ one could think about the 2D convolution in the following way.  
 Think of the 2D signal $ f \left[ m, n \right] $ as a 2D matrix. the output of the convolution for the $ j $ -th element as the $ i $ -th column is basically the result of $ Q $ 1D convolutions of its adjacent columns. Hence it is expected to see some kind of 1D convolution form in the matrix which is indeed composed as following:

 $$ H = \begin{bmatrix}
{T}_{1} & \boldsymbol{0} & \cdots & \boldsymbol{0} & \boldsymbol{0} \\ 
{T}_{2} & {T}_{1} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\ 
\vdots & {T}_{2} & \ddots & \\ 
{T}_{Q} & \vdots & \ddots & \\ 
\boldsymbol{0} & {T}_{Q} &  & \\
\vdots & \boldsymbol{0} & \ddots & \\ 
\boldsymbol{0} &  &  &
\end{bmatrix} $$

The above is Doubly Block Toeptliz Matrix where each matrix $ {T}_{i} $ is a 1D Convolution Toeplitz Matrix of the $ i $ -th column of $ h $.

## Remarks 
 *  The output matrix is composed of $ m \times n $ columns where the $ i $ -th columns corresponds to the output of the convolution between the convolution kernel to a 2D matrix of size $ m \times n $ which all of its elements are zeros but the $ i $ -th element which has the value $ 1 $, namely the $ i $ -th elementary base of $ m \times n $ matrix.
 *  A closely related point of view is given by the [Helix Transform](https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution#The_Helix_Transform).

## References
 *  [How to Generate Block Toeplitz Matrix](https://www.mathworks.com/matlabcentral/answers/249061).
 *  [`FUNC2MAT`- Convert Linear Function into Matrix Form](https://www.mathworks.com/matlabcentral/fileexchange/44669).