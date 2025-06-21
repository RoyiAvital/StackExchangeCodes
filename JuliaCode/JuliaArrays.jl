# StackExchange Code - Julia  Arrays
# Set of functions for Dense Arrays and Sparse Arrays.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.1.001     21/06/2025  Royi Avital RoyiAvital@yahoo.com
#   *   Added `SqueezeArray()` which imitates MATLAB's `squeeze()`.
# - 1.1.000     22/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `ReshapeArray()` as a faster alternative to `reshape()`.
# - 1.0.000     18/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using SparseArrays;

# External

## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

## Functions

function NormalizeRows( mW :: AbstractSparseMatrix{T} ) where {T <: Number}

    numRows = size(mW, 1);
    numCols = size(mW, 2);
    vI, vJ, vV = findnz(mW);
    vRowSum = zeros(numRows);
    numNonZero = length(vI);

    for ii ∈ 1:numNonZero
        # @infiltrate
        vRowSum[vI[ii]] += vV[ii];
    end

    for ii ∈ 1:numRows
        vRowSum[ii] = ifelse(vRowSum[ii] != zero(T), vRowSum[ii], one(T));
    end

    for ii ∈ 1:numNonZero
        vV[ii] /= vRowSum[vI[ii]];
    end

    return sparse(vI, vJ, vV, numRows, numCols);

end

function NormalizeRows( mA :: SparseMatrixCSC{T} ) where {T <: AbstractFloat}

    mW  = copy(mA);
    vRS = Vector{T}(undef, size(mA, 1)); #<! Row Sum

    NormalizeRows!(mW, vRS);

    return mW;

end

function NormalizeRows!( mA :: SparseMatrixCSC{T}, vRowSum :: AbstractVector{T} ) where {T <: AbstractFloat}
    
    vV = nonzeros(mA);
    vR = rowvals(mA); #<! Row index
    vRowSum .= zero(T);
    
    for jj in axes(mA, 2)
        for kk in nzrange(mA, jj)
            ii = vR[kk]; #<! Row index
            vRowSum[ii] += vV[kk];
        end
    end

    for ii ∈ 1:length(vRowSum)
        vRowSum[ii] = ifelse(vRowSum[ii] != zero(T), vRowSum[ii], one(T));
    end
    
    for jj in axes(mA, 2)
        for kk in nzrange(mA, jj)
            ii = vR[kk];
            vV[kk] /= vRowSum[ii];
        end
    end
    
    return mA;

end

function ScaleRowsCols!( mA :: SparseMatrixCSC{T}, vC :: AbstractVector{T}, vR :: AbstractVector{T} ) where {T <: AbstractFloat}
    
    # A[i,j] = r[i] * A[i,j] * c[j]
    # See https://discourse.julialang.org/t/115956
    # TODO: Check and verify!
    
    vI = rowvals(mA);
    vV = nonzeros(mA);
    numCols = size(mA, 2);
    for jj ∈ 1:numCols
        c = vC[jj];
        for ii in nzrange(mA, jj)
            rr = vI[ii];
            vV[ii] *= vR[rr] * c;
        end
    end

    return mA;

end

function ReshapeArray( inArr :: AbstractArray{T, N}, tuDim :: NTuple{M, K} ) where {T, N, K <: Integer, M}

    # @assert (N >= one(N) && (M >= one(M))) "The input and output arrays must have at least one dimension"
    # @assert (length(inArr) == prod(tuDim)) "The number of elements in `inArr` $(length(inArr)) does not match the output number of elements $(prod(tuDim))"
    
    # return Base.__reshape((inArr, IndexLinear()), tuDim);

    return invoke(Base._reshape, Tuple{AbstractArray, typeof(tuDim)}, inArr, tuDim);

end

# See https://discourse.julialang.org/t/14666/21
# ind2sub(tuShape, indices) = Tuple.(CartesianIndices(shape)[indices])
# sub2ind(shape, indices) = LinearIndices(shape)[CartesianIndex.(indices)]

function LinearToSubScripts( tuShape :: NTuple{K, N}, linIdx :: N ) where {N <: Integer, K}
    
    return Tuple(CartesianIndices(tuShape)[linIdx]);

end

function LinearToSubScripts( tuShape :: NTuple{K, N}, vLinIdx :: Vector{N} ) where {N <: Integer, K}
    
    return Tuple.(CartesianIndices(tuShape)[vLinIdx]);

end

function SubScriptsToLinear( tuShape :: NTuple{K, N}, tuSubScriptIdx :: NTuple{K, N} ) where {N <: Integer, K}
    
    return LinearIndices(tuShape)[CartesianIndex(tuSubScriptIdx)];

end

function SubScriptsToLinear( tuShape :: NTuple{2, N}, vSubScriptIdx :: Vector{NTuple{K, N}} ) where {N <: Integer, K}
    
    return LinearIndices(tuShape)[CartesianIndex.(vSubScriptIdx)];

end

function SqueezeArray( mA :: Array )
    
    tuSingletonDim = tuple((d for d in 1:ndims(mA) if size(mA, d) == 1)...);
    return dropdims(mA; dims = tuSingletonDim);

end


