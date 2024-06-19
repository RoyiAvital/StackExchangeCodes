# StackExchange Code - Julia Signal Processing
# Set of functions for Signal Processing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  Add DFT convolution code from https://dsp.stackexchange.com/questions/90036.
# Release Notes
# - 1.0.000     10/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External

## Constants & Configuration

include("./JuliaInit.jl");

## Functions

function PadArray( vA :: Vector{T}, padRadius :: N, padMode :: PadMode; padValue :: T = zero(T) ) where {T <: Real, N <: Signed}
    # TODO: Support padding larger then the input.
    # TODO: Create non allocating variant.
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
    
    I = JJ + K - 1; #<! Output length
	
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
    # L1 Splines for Robust, Simple, and Fast Smoothing of Grid Data (https://arxiv.org/abs/1208.2292)

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