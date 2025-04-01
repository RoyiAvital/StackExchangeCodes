# StackExchange Code - Julia Signal Processing
# Set of functions for Signal Processing.
# References:
#   1.  
# Remarks:
#   1.  Optimize `@inbounds` and `@fastmath` for convolution.
# TODO:
# 	1.  Add DFT convolution code from https://dsp.stackexchange.com/questions/90036.
# 	2.  Remove dependency on `Convex.jl` by using solvers from `JuliaOptimization.jl`.
# Release Notes
# - 1.4.000     26/10/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `OrderFilter()`.
#   *   Added `MedianFilter()`.
#   *   Added `BeadsFilter()`.
#   *   Added packages in use to be explicitly imported.
# - 1.3.000     08/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `GenConvMtx()` for operator view of the 1D convolution.
#   *   Verifying the initialization happens only once.
# - 1.2.000     07/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Made `PadArray()` support any type of `Number`.
# - 1.1.000     03/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `GenGaussianKernel()`.
#   *   Added `PadArray!()`.
#   *   Changed `N` in `PadArray()` into `Integer`.
# - 1.0.000     10/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using SparseArrays;

# External
using Convex;
using ECOS;
using StaticKernels;

## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

## Functions

function PadArray!( vB :: Vector{T}, vA :: Vector{T}, padRadius :: N, padMode :: PadMode; padValue :: T = zero(T) ) where {T <: Number, N <: Integer}
    # TODO: Support padding larger then the input.
    # TODO: Add modes: `pre`, `after`, `both`.

    numElementsA = length(vA);
    numElementsB = numElementsA + 2padRadius

    vB = Vector{T}(undef, numElementsB);
    
    if (padMode == PAD_MODE_CONSTANT) 
        vB .= padValue;
        vB[(padRadius + 1):(numElementsB - padRadius)] .= vA;
    end

    if (padMode == PAD_MODE_REPLICATE)
        for ii in 1:numElementsB
            jj = clamp(ii - padRadius, 1, numElementsA);
            vB[ii] = vA[jj];
        end
    end

    if (padMode == PAD_MODE_SYMMETRIC)
        for ii in 1:numElementsB
            jj = ifelse(ii - padRadius < 1, padRadius - ii + 1, ii - padRadius);
            jj = ifelse(ii - padRadius > numElementsA, numElementsA - (ii - (numElementsA + padRadius) - 1), jj);
            vB[ii] = vA[jj];
        end
    end

    if (padMode == PAD_MODE_REFLECT) #<! Mirror  [c, b][a, b, c][b, a]
        for ii in 1:numElementsB
            jj = ifelse(ii - padRadius < 1, padRadius - ii + 2, ii - padRadius);
            jj = ifelse(ii - padRadius > numElementsA, numElementsA - (ii - (numElementsA + padRadius)), jj);
            vB[ii] = vA[jj];
        end
    end

    if (padMode == PAD_MODE_CIRCULAR)
        for ii in 1:numElementsB
            jj = ifelse(ii - padRadius < 1, numElementsA + ii - padRadius, ii - padRadius);
            jj = ifelse(ii - padRadius > numElementsA, ii - numElementsA - padRadius, jj);
            vB[ii] = vA[jj];
        end
    end

    return vB;

end

function PadArray( vA :: Vector{T}, padRadius :: N, padMode :: PadMode; padValue :: T = zero(T) ) where {T <: Number, N <: Integer}
    # TODO: Support padding larger then the input.
    # TODO: Add modes: `pre`, `after`, `both`.

    numElementsA = length(vA);
    numElementsB = numElementsA + 2padRadius

    vB = Vector{T}(undef, numElementsB);
    
    PadArray!(vB, vA, padMode; padValue = padValue);

    return vB;

end

function Conv1D( vA :: Vector{T}, vB :: Vector{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: Real}
    
    if (convMode == CONV_MODE_FULL)
        vO = Vector{T}(undef, length(vA) + length(vB) - 1);
    elseif (convMode == CONV_MODE_SAME)
        vO = Vector{T}(undef, length(vA));
    elseif (convMode == CONV_MODE_VALID)
        vO = Vector{T}(undef, length(vA) - length(vB) + 1);
    end

    Conv1D!(vO, vA, vB; convMode);
    return vO;

end

function Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: Real}
    
    if (convMode == CONV_MODE_FULL)
        _Conv1D!(vO, vA, vB);
    elseif (convMode == CONV_MODE_SAME)
        _Conv1DSame!(vO, vA, vB);
    elseif (convMode == CONV_MODE_VALID)
        _Conv1DValid!(vO, vA, vB);
    end

end

function _Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Full convolution.
    # Seems that `@simd` is much worse than `@turbo`.
    # Keeps `@simd` as it does not require dependency.

    J = length(vA);
    K = length(vB); #<! Assumed to be the Kernel
    
    # Optimized for the case the kernel is in vB (Shorter)
    J < K && return _Conv1D!(vO, vB, vA);
    
    I = J + K - 1; #<! Output length
	
    @simd ivdep for ii in 1:(K - 1) #<! Head
        sumVal = zero(T);
        for kk in 1:K #<! Don't make kk depends on ii!
            ib0 = (ii >= kk);
            @inbounds oa = ib0 ? vA[ii - kk + 1] : zero(T);
            @inbounds sumVal += vB[kk] * oa;
        end
        @inbounds vO[ii] = sumVal;
    end
    @simd ivdep for ii in K:(J - 1) #<! Middle
        sumVal = zero(T);
        for kk in 1:K
            @inbounds sumVal += vB[kk] * vA[ii - kk + 1];
        end
        @inbounds vO[ii] = sumVal;
    end
    @simd ivdep for ii in J:I #<! Tail
        sumVal = zero(T);
        for kk in 1:K
            ib0 = (ii < J + kk);
            @inbounds oa = ib0 ? vA[ii - kk + 1] : zero(T);
            @inbounds sumVal += vB[kk] * oa;
        end
        @inbounds vO[ii] = sumVal;
    end

end

function _Conv1DSame!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Same convolution.
    # Seems that `@simd` is much worse than `@turbo`.
    # Keeps `@simd` as it does not require dependency.
    # TODO: Support the case K >= J.

    J = length(vA);
    K = length(vB); #<! Assumed to be the Kernel

    R = floor(Int, K / 2);
    F = 1 + R;
    
    # Optimized for the case the kernel is in vB (Shorter)
    # J < K && return _Conv1DSame!(vO, vB, vA);
    
    I = J; #<! Output length
	
    @simd ivdep for ii in 1:(K - 1) #<! Head
        sumVal = zero(T);
        for kk in 1:K #<! Don't make kk depends on ii!
            ib0 = (ii + R >= kk);
            @inbounds oa = ib0 ? vA[ii - kk + R + 1] : zero(T);
            @inbounds sumVal += vB[kk] * oa;
        end
        @inbounds vO[ii] = sumVal;
    end
    @simd ivdep for ii in F:(J - R) #<! Middle
        sumVal = zero(T);
        for kk in 1:K
            @inbounds sumVal += vB[kk] * vA[ii - kk + R + 1];
        end
        @inbounds vO[ii] = sumVal;
    end
    @simd ivdep for ii in (J - R + 1):I #<! Tail
        sumVal = zero(T);
        for kk in 1:K
            ib0 = (R - (J - ii) < kk);
            @inbounds oa = ib0 ? vA[ii - kk + R + 1] : zero(T);
            @inbounds sumVal += vB[kk] * oa;
        end
        @inbounds vO[ii] = sumVal;
    end

end

function _Conv1DValid!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Valid convolution.
    # Seems that `@simd` is much worse than `@turbo`.
    # Keeps `@simd` as it does not require dependency.
    # TODO: Add support for `same` and `valid` modes.

    J = length(vA);
    K = length(vB); #<! Assumed to be the Kernel
    
    # If kernel is smaller than signal it is not defined
    J < K && return;
    
    I = J - K + 1; #<! Output length

    @simd ivdep for ii in 1:I #<! Middle
        sumVal = zero(T);
        for kk in 1:K
            @inbounds sumVal += vB[kk] * vA[ii - kk + K];
        end
        @inbounds vO[ii] = sumVal;
    end

end

function GenConvMtx( vK :: AbstractVector{T}, numElements :: N; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat, N <: Integer}
    # Convolution matrix for 1D convolution.
    
    kernelLength = length(vK);

    if (convMode == CONV_MODE_FULL)
        rowIdxFirst = 1;
        rowIdxLast  = numElements + kernelLength - 1;
        outputSize  = numElements + kernelLength - 1;
    elseif (convMode == CONV_MODE_SAME)
        rowIdxFirst = 1 + floor(kernelLength / 2);
        rowIdxLast  = rowIdxFirst + numElements - 1;
        outputSize  = numElements;
    elseif (convMode == CONV_MODE_VALID)
        rowIdxFirst = kernelLength;
        rowIdxLast  = (numElements + kernelLength - 1) - kernelLength + 1;
        outputSize  = numElements - kernelLength + 1;
    end

    mtxIdx = 0;
    vI = ones(N, numElements * kernelLength);
    vJ = ones(N, numElements * kernelLength);
    vV = zeros(T, numElements * kernelLength);

    for jj ∈ 1:numElements
        for ii ∈ 1:kernelLength
            if((ii + jj - 1 >= rowIdxFirst) && (ii + jj - 1 <= rowIdxLast))
                # Valid output matrix row index
                mtxIdx += 1;
                vI[mtxIdx] = ii + jj - rowIdxFirst;
                vJ[mtxIdx] = jj;
                vV[mtxIdx] = vK[ii];
            end
        end
    end

    mK = sparse(vI, vJ, vV, outputSize, numElements);

    return mK;

end

function L1Spline( vY :: Vector{T}, λ :: T ) where {T <: Real}
    # L1 Splines for Robust, Simple, and Fast Smoothing of Grid Data (https://arxiv.org/abs/1208.2292)

    numSamples = length(vY);

    mD = spdiagm(numSamples, numSamples, -1 => ones(numSamples - 1), 0 => -2 * ones(numSamples), 1 => ones(numSamples - 1));
    mD = mD[2:(end - 1), :];

    vX = Variable(numSamples);
    sConvProb = minimize( norm(vX - vY, 1) + λ * sumsquares(mD * vX) );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    vZ = vec(vX.value);

    return vZ;

end

function L1PieceWise( vY :: Vector{T}, λ :: T, polyDeg :: N; ρ :: T = 1.0 ) where {T <: Real, N <: Integer}
    # Piece Wise Model with Auto Know Selection: https://dsp.stackexchange.com/questions/1227.

    numSamples = length(vY);

    mD = spdiagm(numSamples, numSamples, 0 => -ones(numSamples), 1 => ones(numSamples - 1));
    mDD = copy(mD);
    for kk in 1:polyDeg
        mD[:] = mD * mDD;
    end
    mD = mD[1:(end - polyDeg - 1), :];

    vX = Variable(numSamples);
    sConvProb = minimize( 0.5 * sumsquares(vX - vY) + λ * norm(mD * vX, 1) );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    vZ = vec(vX.value);

    return vZ;

end

function GenGaussianKernel( σ :: T, kernelRadius :: N ) where {T <: AbstractFloat, N <: Integer}

    numElements = N(2) * kernelRadius + 1;

    vK = zeros(T, numElements);

    for (ii, mm) ∈ enumerate(-kernelRadius:kernelRadius)
        vK[ii] = exp(- (mm * mm) / (T(2.0) * σ * σ));
    end

    vK ./= sum(mK);

    return vK;

end

function MedianFilter( vX :: Vector{T}, localRadius :: N ) where {T <: AbstractFloat, N <: Integer}
    # Should be used with a small radii.
    
    # https://github.com/stev47/StaticKernels.jl/discussions/12
    vK = Kernel{(-localRadius:localRadius, )}(@inline w -> median(Tuple(w)));
    vY = map(vK, extend(vX, StaticKernels.ExtensionSymmetric()));
    
    return vY;

end

function OrderFilter( vX :: Vector{T}, localRadius :: N, k :: N ) where {T <: AbstractFloat, N <: Integer}
    # Should be used with a small radii.

    numElements = (N(2) * localRadius[1]) + one(N);
    vB = zeros(T, numElements);

    # Ascending order
    vK = Kernel{(-localRadius:localRadius, )}(@inline w -> (copyto!(vB, Tuple(w)); sort!(vB)[k]));
    vY = map(vK, extend(vX, StaticKernels.ExtensionSymmetric()));
    
    return vY;

end

function BeadsFilter( vY :: Vector{T}, modelDeg :: N, fₛ :: T, asyRatio :: T, λ₀ :: T, λ₁ :: T, λ₂ :: T, numIter :: N; ϵ₀ :: T = T(1e-6), ϵ₁ :: T = T(1e-6) ) where {T <: AbstractFloat, N <: Integer}
    """
    BeadsFilter(vY::Vector{T}, modelDeg::N, fₛ::T, asyRatio::T, λ₀::T, λ₁::T, λ₂::T, numIter::N; ϵ₀::T = T(1e-6), ϵ₁::T = T(1e-6)) where {T <: AbstractFloat, N <: Integer}
    
    BEADS: Baseline Estimation And Denoising with Sparsity
    
    This function performs baseline estimation and denoising on a signal `vY` using sparsity.
    It is based on the BEADS algorithm, commonly used for processing noisy signals with asymmetric peaks.
    BEADS provides a smooth baseline estimation while preserving the peak information.  
    The algorithm is optimized for a train of positive peaks.
    The BEADS algorithm solves a Convex problem using Majorization Minimization. 
    
    Arguments:
      - `vY::Vector{T}`: The input signal as a vector of type `T`.
      - `modelDeg::N`: Degree of the polynomial model (integer).
      - `fₛ::T`: Cutoff frequency for the baseline filter. In the range [0, 0.5].  
        Commonly in the range `[0, 0.05]`.
      - `asyRatio::T`: Asymmetry ratio for penalizing peaks of different polarity.
         Higher value promotes more positive peaks. Range in (0, ∞).
      - `λ₀::T`: Regularization parameter for baseline sparsity.
      - `λ₁::T`: Regularization parameter for the first order difference (Smoothness).
      - `λ₂::T`: Regularization parameter for the second order difference (Smoothness).
      - `numIter::N`: Number of iterations to run the optimization process.
      
    Keyword Arguments:
      - `ϵ₀::T = T(1e-6)`: Small threshold for handling very small values in the asymmetry constraint.
      - `ϵ₁::T = T(1e-6)`: Small threshold for handling very small values in the sparsity function.
      
    Returns:
      - `vX::Vector{T}`: Filtered signal (Denoised version of `vY` with baseline removed).
      - `vF::Vector{T}`: Estimated baseline signal.
      - `vC::Vector{T}`: Cost function values at each iteration, indicating convergence.
    
    Details:
    The BEADS algorithm operates by balancing between sparsity and smoothness constraints to isolate the baseline from the signal.  
    It includes the following main steps:
    
      1. **Asymmetry-based Sparsity**: The function `θ` penalizes signal values based on an asymmetry ratio (`asyRatio`). It discourages peaks in the baseline, adapting for positive and negative peaks differently.
      2. **Regularization with Sparsity and Smoothness**: 
        - The first-order difference term (`λ₁`) penalizes rapid changes, helping smooth the baseline.
        - The second-order difference term (`λ₂`) penalizes curvature, maintaining a smoother baseline profile.
      3. **Model Matrix Generation**: `GenModelMat` creates the system matrices `mA` and `mB` for the polynomial model specified by `modelDeg`.
      4. **Iterative Optimization**: The algorithm iteratively minimizes a cost function, `hCost`, balancing baseline smoothness and signal sparsity.
    
    Remarks:
      - Results differ from MATLAB (Small values, RMSE of ~0.03).
      - Works well when the signal goes to zero on both ends.  
        A trick to achieve it is by padding a smooth roll to 0.  
        One way to achieve smooth roll is by using Sigmoid.    
    
    Example Usage:
    ```julia
    # Sample noisy signal
    vY = rand(Float64, 100) .+ sin.(range(0, 2π, length = 100));
    
    # Parameters
    modelDeg = 2;
    fₛ = 0.05;
    asyRatio = 0.5;
    λ₀ = 1.0;
    λ₁ = 1.0;
    λ₂ = 1.0;
    numIter = 100;
    
    # Run BEADS filter
    vX, vF, vC = BeadsFilter(vY, modelDeg, fₛ, asyRatio, λ₀, λ₁, λ₂, numIter);

    Reference:
      - Chromatogram baseline estimation and denoising using sparsity (BEADS).
    """

    ϕ(vY :: Vector{T}) = abs.(vY) .- ϵ₁ .* log.(abs.(vY) .+ ϵ₁); #<! Differentiable approximation of L1
    ψ(vY :: Vector{T}) = inv.(abs.(vY) .+ ϵ₁); #<! wfun on MATLAB
    
    function θ(vY :: Vector{T}) # where {T <: AbstractFloat} 
        valSum = zero(T);

        for ii ∈ 1:length(vY)
            valSum += ifelse(vY[ii] > ϵ₀, vY[ii], zero(T));
            valSum += ifelse(vY[ii] < -ϵ₀, -asyRatio * vY[ii], zero(T));
            valSum += ifelse(abs(vY[ii]) < ϵ₀, ((one(T) + asyRatio) / (T(4)ϵ₀)) * vY[ii] * vY[ii], zero(T));
            valSum += ifelse(abs(vY[ii]) < ϵ₀, ((one(T) - asyRatio) / T(2)) * vY[ii] + (ϵ₀ * (one(T) + asyRatio) / T(4)), zero(T));
        end

        return valSum;
    end
    
    function GenModelMat(modelDeg :: N, fₛ :: T, numSamples :: N) # where {T <: AbstractFloat, N <: Integer} 

        vB1 = [one(T), -one(T)];
        vB2 = [-one(T), T(2), -one(T)];
        for _ ∈ one(N):(modelDeg - one(N))
            vB1 = Conv1D(vB1, vB2);
        end
        vB = Conv1D(vB1, [-one(T), one(T)]);

        valArg = 2π * fₛ;
        valT = ((one(T) - cos(valArg)) / (one(T) + cos(valArg))) ^ modelDeg;

        vA = ones(T, 1);
        vA1 = [one(T), T(2), one(T)];
        for _ ∈ one(N):modelDeg
            vA = Conv1D(vA, vA1);
        end

        vA = vB + valT * vA;

        vDiag = [dd => vA[ii] * ones(T, numSamples - abs(dd)) for (ii, dd) ∈ enumerate(-modelDeg:modelDeg)];
        mA = spdiagm(numSamples, numSamples, vDiag...);

        vDiag = [dd => vB[ii] * ones(T, numSamples - abs(dd)) for (ii, dd) ∈ enumerate(-modelDeg:modelDeg)];
        mB = spdiagm(numSamples, numSamples, vDiag...);

        return mA, mB;
        
    end
    
    vX = copy(vY);
    vC = zeros(T, numIter); #<! Cost per iteration
    numSamples = length(vY);

    mA, mB = GenModelMat(modelDeg, fₛ, numSamples);

    hH(vX :: Vector{T}) = mB * (mA \ vX);
    hCost(vX :: Vector{T}) = T(0.5) * sum(abs2, hH(vY - vX)) + λ₀ * θ(vX) + λ₁ * sum(ϕ(diff(vX))) + λ₂ * sum(ϕ(diff(diff(vX))));
    vE = ones(T, numSamples - one(N));
    mD1 = spdiagm(numSamples - one(N), numSamples, 0 => -vE, 1 => vE);
    mD2 = spdiagm(numSamples - N(2), numSamples, 0 => vE[1:(end - 1)], 1 => -T(2) * vE[1:(end - 1)], 2 => vE[1:(end - 1)]);

    mD = cat(mD1, mD2; dims = 1);
    mBB = mB' * mB;

    vW = cat(λ₁ * ones(T, numSamples - one(N)), λ₂ * ones(T, numSamples - N(2)); dims = 1);
    mΛ = spdiagm(vW .* ψ(mD * vX)); #<! Shape (2 * numSamples - 3, 2 * numSamples - 3)
    vB = T(0.5) * (one(T) - asyRatio) * ones(numSamples);
    vD = mBB * (mA \ vY) - λ₀ * (mA' * vB);
    vΓ = ones(T, numSamples);
    mΓ = spdiagm(vΓ); #<! TODO: Check using Diagonal

    vC[1] = hCost(vX);

    for ii ∈ N(1):numIter
        mΛ.nzval[:] = vW .* ψ(mD * vX);
        for jj ∈ 1:numSamples
            vΓ[jj] = ((one(T) + asyRatio) / T(4)) / ifelse(abs(vX[jj]) > ϵ₀, abs(vX[jj]), ϵ₀);
        end

        mΓ.nzval[:] = vΓ[:]; #<! Update diagonal
        mM = T(2) * λ₀ * mΓ + (mD' * mΛ * mD); #<! TODO: Optimize (No need for `mΛ`)
        vX = mA * ((mBB + (mA' * mM * mA)) \ vD);

        vC[ii] = hCost(vX);
    end

    vF = vY - vX - hH(vY - vX);
    
    return vX, vF, vC;

end


