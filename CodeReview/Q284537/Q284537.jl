# StackExchange Code Review Q284537
# https://codereview.stackexchange.com/questions/284537
# Implementing a 1D Convolution SIMD Friendly in Julia
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     28/04/2023  Royi Avital
#   *   First release.

## Packages

using LinearAlgebra;

# using Plots;
# using Printf;
using BenchmarkTools;
using LoopVectorization;
using StaticKernels;
using PaddedViews;

## Constants

## General Parameters

figureIdx = 0;

## Functions

# Generic Wrapper 
function Conv1DWrapper( vA :: Array{T, 1}, vB :: Array{T, 1}, hConvFun ) :: Array{T, 1} where {T <: Real}

    lenA = length(vA);
    lenB = length(vB);

    vO = Array{T, 1}(undef, lenA + lenB - 1);

    return hConvFun(vO, vA, vB);

end

# Reference
function Conv1D!( vO :: Array{T, 1}, vA :: Array{T, 1}, vB :: Array{T, 1} ) :: Array{T, 1} where {T <: Real}

    lenA = length(vA);
    lenB = length(vB);
    lenO = length(vO);
    vC   = view(vB, lenB:-1:1);
    @simd for ii in 1:lenO
        # Rolling vB over vA
        startIdxA = max(1, ii - lenB + 1);
        endIdxA   = min(lenA, ii);
        startIdxC = max(lenB - ii + 1, 1);
        endIdxC   = min(lenB, lenO - ii + 1);
        # println("startA = $startIdxA, endA = $endIdxA, startC = $startIdxC, endC = $endIdxC");
        @inbounds vO[ii] = LinearAlgebra.dot(view(vA, startIdxA:endIdxA), view(vC, startIdxC:endIdxC));
    end

    return vO;

end

# Vanilla Code
function _Conv1D!( vO :: Array{T, 1}, vA :: Array{T, 1}, vB :: Array{T, 1} ) :: Array{T, 1} where {T <: Real}

    lenA = length(vA);
    lenB = length(vB);

    fill!(vO, zero(T));
    for idxB in 1:lenB
        @simd for idxA in 1:lenA
            @inbounds vO[idxA + idxB - 1] += vA[idxA] * vB[idxB];
        end
    end

    return vO;

end

# From Discourse by mikmoore
function __Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # From https://discourse.julialang.org/t/97658/2 (conv1y!)
    # A bit slower than _Conv1D!()
	
    fill!(vO, zero(T));
	
    @inbounds for idxB in eachindex(vB)
		@simd for idxA in intersect(eachindex(vA), eachindex(vO) .- (idxB - 1))
			vO[idxB - 1 + idxA] += vA[idxA] * vB[idxB]
		end
	end
	return vO
end

# From Discourse by Elrod
function ___Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Based on https://discourse.julialang.org/t/97658/15
    # Doesn't require `StrideArraysCore` (Uses 1 based index vs. 0 based index)

    J = length(vA);
    K = length(vB); #<! Assumed to be the Kernel
    
    # Optimized for the case the kernel is in vB (Shorter)
    J < K && return ___Conv1D!(vO, vB, vA);
    
    I = J + K - 1; #<! Output length
	
    @turbo for ii in 0:(K - 1) #<! Head
        sumVal = zero(T);
        for kk in 1:K
            ib0 = (ii >= kk);
            oa = ib0 ? vA[ii - kk + 1] : zero(T);
            sumVal += vB[kk] * oa;
        end
        vO[ii] = sumVal;
    end
    @turbo inline=true for ii in K:(J - 1) #<! Middle
        sumVal = zero(T);
        for kk in 1:K
            sumVal += vB[kk] * vA[ii - kk + 1];
        end
        vO[ii] = sumVal;
    end
    @turbo for ii in J:I #<! Tail
        sumVal = zero(T);
        for kk in 1:K
            ib0 = (ii < J + kk);
            oa = ib0 ? vA[ii - kk + 1] : zero(T);
            sumVal += vB[kk] * oa;
        end
        vO[ii] = sumVal;
    end
	return vO
end

function ____Conv1D!( vO :: Vector{T}, vA :: Vector{T}, vB :: Vector{T} ) where {T <: Real}
    # Seems that `@simd` is much worse than `@turbo`.

    II = length(vA);
    K = length(vB); #<! Assumed to be the Kernel
    
    # Optimized for the case the kernel is in vB (Shorter)
    J < K && return ___Conv1D!(vO, vB, vA);
    
    I = II + K - 1; #<! Output length
	
    @turbo for ii in 1:(K - 1) #<! Head
        sumVal = zero(T);
        for kk in 1:K
            ib0 = (ii >= kk);
            @inbounds oa = ib0 ? vA[ii - kk + 1] : zero(T);
            @inbounds sumVal += vB[kk] * oa;
        end
        @inbounds vO[ii] = sumVal;
    end
    @turbo inline=true for ii in K:(J - 1) #<! Middle
        sumVal = zero(T);
        for kk in 1:K
            @inbounds sumVal += vB[kk] * vA[ii - kk + 1];
        end
        @inbounds vO[ii] = sumVal;
    end
    @turbo for ii in J:I #<! Tail
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

# function ___Conv1D!(c::AbstractVector{T}, a::AbstractVector{T}, b::AbstractVector{T}) where {T}
#   I = length(c)
#   K = length(b)
#   J = I - K + 1
#   J < K && return ___Conv1D!(c, b, a)
#   @turbo for i = 1:K-1
#     s = zero(T)
#     for k = 1:K
#       ib0 = (i >= k)
#       oa = ib0 ? a[i - k + 1] : zero(T)
#       s += b[k] * oa
#     end
#     c[i] = s
#   end
#   @turbo inline=true for i = K:J-1
#     s = zero(T)
#     for k = 1:K
#       s += b[k] * a[i - k + 1]
#     end
#     c[i] = s
#   end
#   @turbo for i = J:I
#     s = zero(T)
#     for k = 1:K
#       ib0 = (i < J + k)
#       oa = ib0 ? a[i - k + 1] : zero(T)
#       s += b[k] * oa
#     end
#     c[i] = s
#   end
# end

# using LoopVectorization, StrideArraysCore
# using StrideArraysCore: static, static_length, zero_offsets

# function conv3!(
#   _c::AbstractVector{T},
#   _a::AbstractVector{T},
#   _b::AbstractVector{T},
# ) where {T}
#   c = zero_offsets(_c)
#   a = zero_offsets(_a)
#   b = zero_offsets(_b)
#   I = static_length(c)
#   K = static_length(b)
#   J = I - K + static(1)
#   J < K && return conv3!(_c, _b, _a)
#   @turbo for i = 0:K-2
#     s = zero(T)
#     for k = 0:K-1
#       ib0 = (i >= k)
#       oa = ib0 ? a[i-k] : zero(T)
#       s += b[k] * oa
#     end
#     c[i] = s
#   end
#   @turbo inline=true for i = K-1:J-1
#     s = zero(T)
#     for k = 0:K-1
#       s += b[k] * a[i-k]
#     end
#     c[i] = s
#   end
#   @turbo for i = J:I-1
#     s = zero(T)
#     for k = 0:K-1
#       ib0 = (i < J+k)
#       oa = ib0 ? a[i-k] : zero(T)
#       s += b[k] * oa
#     end
#     c[i] = s
#   end
# end

vA = [1, 2, 3, 4, 5];
vB = [4, 5, 6];
vC = [6, 5, 4];
vO = collect(1:(length(vA) + length(vB) - 1));

# numSamplesA = 1000;
# numSamplesB = 3;

# vA = rand(numSamplesA);
# vB = rand(numSamplesB);
# vO = rand(numSamplesA + numSamplesB - 1);

# conv3!(vO, vA, vB);

___Conv1D!(vO, vA, vB);

# vC = view(vB, length(vB):-1:1);
# vK = Kernel{(0:(length(vB) - 1), )}(@inline vW -> sum(vW .* vC));
# vK = Kernel{(1:length(vB), )}(@inline vW -> sum(vW .* vC));
# vK = @kernel w -> w[0] * vB[3] + w[1] * vB[2] + w[2] * vB[1];
# vK = Kernel{(0:(length(vB) - 1), )}(@inline vW -> vW[0] * vB[3] + vW[1] * vB[2] + vW[2] * vB[1]);
# vK = Kernel{(-1:1, )}(@inline vW -> vW[0] * vB[3] + vW[1] * vB[2] + vW[2] * vB[1]);
# map(vK, extend(vA, StaticKernels.ExtensionConstant(0)));
# map!(vK, vO, extend(vA, StaticKernels.ExtensionConstant(0)));

# numElementsPad = length(vA) + (2 * (length(vB) - 1));
# vParnetIdx = (length(vB) - 1) .+ (1:length(vA));
# vAA = PaddedView(0, vA, (1:numElementsPad,), (vParnetIdx,)); #<! View
# vK = Kernel{(0:(length(vB) - 1), )}(@inline vW -> vW[0] * vB[3] + vW[1] * vB[2] + vW[2] * vB[1]);
# vK = Kernel{(0:(length(vB) - 1), )}(@inline vW -> vW[0] * vB[15] + vW[1] * vB[14] + vW[2] * vB[13] + vW[3] * vB[12] + vW[4] * vB[11] + vW[5] * vB[10] + vW[6] * vB[9] + vW[7] * vB[8] + vW[8] * vB[7] + vW[9] * vB[6] + vW[10] * vB[5] + vW[11] * vB[4] + vW[12] * vB[3] + vW[13] * vB[2] + vW[14] * vB[1]);
# map!(vK, vO, vAA);
# vO

# @benchmark Conv1D!($vO, $vA, $vB)
# @benchmark _Conv1D!($vO, $vA, $vB)
# @benchmark __Conv1D!($vO, $vA, $vB)
# @benchmark ___Conv1D!($vO, $vA, $vB)
# @benchmark ____Conv1D!($vO, $vA, $vB)
# @benchmark conv3!($vO, $vA, $vB)
# @benchmark map!($vK, $vO, $vAA)
