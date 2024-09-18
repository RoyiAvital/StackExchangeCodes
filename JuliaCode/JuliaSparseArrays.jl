# StackExchange Code - Julia Sparse Arrays
# Set of functions for Sparse Arrays.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
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




