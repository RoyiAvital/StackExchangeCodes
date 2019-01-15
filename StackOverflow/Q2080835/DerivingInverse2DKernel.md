# Deriving the Inverse Kernel of a Given 2D Convolution Kernel

This is basically a generalization of the question - [Deriving the Inverse Filter of Image Convolution Kernel](https://stackoverflow.com/questions/2080835).


## Problem Formulation

Given a Convolution Kernel $ f \in \mathbb{R}^{m \times n} $ find its inverse kernel, $ g \in \mathbb{R}^{p \times q} $ such that $ f \ast g = h = \delta $.

## Solution I

One could build the Matrix Form of the Convolution Operator.  
The Matrix form can replicate various mode of Image Filtering (Exclusive):  

 *	Boundary Conditions (Applied as padding the image and apply Convolution in Valid mode).
 	*	Zero Padding
 		Padding the image with zeros.
 	*	Circular  
 		Using Circular / Periodic continuation of the image.
 		Match Frequency Domain convolution.
 	*	Replicate  
 		Replicating the edge values (Nearest Neighbor).  
 		Usually creates the least artifacts in real world.
 	*	Symmetric  
 		Mirroring the image along the edges.
 *	Convolution Shape
 	*	Full
 		The output size is to full extent - $ \left( m + p - 1 \right) \times \left( n + q - 1 \right) $.
 	*	Same  
 		Output size is the same as the input size ("Image").
 	*	Valid  
 		Output size is the size of full overlap of the image and kernel.

When using Image Filtering the output size matches the input size hence the Matrix Form is square and the inverse is defined.  
Using Convolution Matrix the matrix might not be square (Unless "Same" is chosen) hence the Pseudo Inverse should be derived.  

One should notice that while the input matrix should be sparse the inverse matrix isn't.  
Moreover while the Convolution Matrix will have special form (Toeplitz neglecting the Boundary Conditions) the inverse won't be.  
Hence this solution is more accurate than the next one yet it also use higher degree of freedom (The solution isn't necessarily in Toeplitz form).

## Solution II

In this solution the inverse is derived using minimization of the following cost function:

$$ \arg \min_{g} \frac{1}{2} {\left\| f \ast g - h \right\|}_{2}^{2} $$

The derivative is given by:

$$ \frac{\partial \frac{1}{2} {\left\| f \ast g - h \right\|}_{2}^{2} }{\partial g} = f \star \left( f \ast g - h \right) $$

Where $ \star $ is the Correlation operation.

In practice the convolution in the Objective Function is done in `full` mode (MATLAB idiom).  
In practice it is done:

```matlab
hObjFun 	= @(mG) 0.5 * sum((conv2(mF, mG, 'full') - mH) .^ 2, 'all');
mObjFunGrad = conv2(conv2(mF, mG, 'full') - mH, mF(end:-1:1, end:-1:1), 'valid');
```

Where the code flips `mF` for Correaltion and use valid (In matrix form it is the Adjoint / Transpose -> the output is smaller).

The optimization problem is Strictly Convex and can be solved easily using Gradient Descent:

```matlab
for ii = 1:numIteraions
    mObjFunGrad = conv2(conv2(mF, mG, 'full') - mH, mF(end:-1:1, end:-1:1), 'valid');
    mG          = mG - (stepSize * mObjFunGrad);
end
```