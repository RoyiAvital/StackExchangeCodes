# Show That the Power Spectrum Density Matrix Is Positive Semi Definite (PSD) Matrix

Given a Wide Sense Stationary Multi Variate Random Process $ \boldsymbol{x} \left[ n \right] $ it Auto Covariance **Matrix** Function is given by:

$$ {R}_{x, x} \left[ m \right] = \mathbb{E} \left[ \boldsymbol{x} \left[ n \right] \boldsymbol{x}^{T} \left[ n - m \right] \right] $$

Prove that the Power Spectrum Density **Matrix** is Positive Semi Definite (PSD) Matrix where it is given by:

$$ {S}_{x, x} \left( f \right) = \sum_{m = -\infty}^{\infty} {R}_{x, x} \left[ m \right] {e}^{-j 2 \pi f m} $$

### Remark
Pay attention that $ {R}_{x, x} \left[ m \right] $ isn't necessarily Positive Semi Definite matrix (It is for $ m = 0 $, yet nothing can be said in general for other time indices). It is even not necessarily  symmetric.

# Proof

Pay attention that for Scalar Random Process the Power Spectrum Density is non negative.

Namely, let $ y \left[ n \right] \in \mathbb{R} $ a WSS Random process with its Auto Correlation function given by:

$$ {R}_{y, y} \left[ m \right] = \mathbb{E} \left[ y \left[ n \right] y \left[ n - m \right] \right] $$

Then the Power Spectrum Denisty is:

$$ {S}_{y, y} \left( f \right) = \sum_{m = -\infty}^{\infty} {R}_{y, y} \left[ m \right] {e}^{-j 2 \pi f k} \geq 0 $$

Then, by defining $ z \left( t \right) = \boldsymbol{v}^{T} \boldsymbol{x} $ one would get:

$$ \begin{align*}
0 \leq {S}_{z, z} \left( f \right) & =  \sum_{m = -\infty}^{\infty} \mathbb{E} \left[ z \left[ n \right] {z}^{T} \left[ n - m \right] \right] {e}^{-j 2 \pi f k} \\
& = \sum_{k = -\infty}^{\infty} \mathbb{E} \left[ \boldsymbol{v}^{T} \boldsymbol{x} \left[ n \right] \boldsymbol{x}^{T} \left[ m - m \right] \boldsymbol{v} \right] {e}^{-j 2 \pi f m} \\
& = \boldsymbol{v}^{T} \left( \mathbb{E} \left[ \boldsymbol{x} \left[ n \right] \boldsymbol{x}^{T} \left[ n - m \right] \right] {e}^{-j 2 \pi f m} \right) \boldsymbol{v} \\
& = \boldsymbol{v}^{T} {S}_{x, x} \left( f \right) \boldsymbol{v} \\
& \Rightarrow {S}_{x, x} \left( f \right) \succeq 0
\end{align*} $$

Some remarks regarding simulating it in MATLAB:

 * If one use MATLAB's `xcorr()` to calculate the Auto Correlation one should use `iffstshift()` to shift the function to be "Symmetric" to MATLAB in order to have the DFT Real and Non Negative. This is due to the fast MATLAB's `fft()` expects the first sample to be of index $ 0 $ (See the `lags` output of `xcorr()`).