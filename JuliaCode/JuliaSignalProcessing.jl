# StackExchange Code - Julia Signal Processing
# Set of functions for Signal Processing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     10/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal

# External
using ColorTypes;

## Constants & Configuration

include("./JuliaInit.jl");

## Functions

function Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T}; convMode :: ConvMode = FULL ) where {T <: Real}
    
    if (convMode == FULL)
        _Conv1D!(vO, vA, vB);
    elseif (convMode == SAME)
        _Conv1DSame!(vO, vA, vB);
    elseif (convMode == SAME)
        _Conv1DValid!(vO, vA, vB);
    end

end

function _Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Full convolution.
    # Seems that `@simd` is much worse than `@turbo`.
    # Keeps `@simd` as it does not require dependency.
    # TODO: Add support for `same` and `valid` modes.

    J = length(vA);
    K  = length(vB); #<! Assumed to be the Kernel
    
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
	return vO
end

function _Conv1DSame!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Full convolution.
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
	return vO
end

function _Conv1DValid!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Full convolution.
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