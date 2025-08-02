# StackExchange Code - Julia Linear Algebra
# Set of functions for Sparse Arrays.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.1.000     02/08/2025  Royi Avital RoyiAvital@yahoo.com
#   *   Removed `LoopVectorization.jl`.
#   *   Added `Adjugate()`.
# - 1.0.000     18/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External

## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

## Functions

function QuadForm( vX :: Vector{T}, mA :: Matrix{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    (axes(vX)..., axes(vY)...) == axes(mA) || throw(DimensionMismatch());
    m, n = size(mA);
    s = zero(T);
    @fastmath for jj in 1:n
        @inbounds yj = vY[jj];
        t = zero(T);
        @simd for ii in 1:m
            @inbounds t += mA[ii, jj] * vX[ii];
        end
        s += t * yj;
    end

    return s;

end

function QuadForm( vX :: Vector{T}, mA :: S, vY :: Vector{T} ) where {T <: AbstractFloat, S <: Symmetric{<: T, <: Matrix{<: T}}}
    # Slower than not using the Symmetric property

    (length(vX) == length(vY) == size(mA, 1)) || throw(DimensionMismatch())
    n = length(vY);
    s = zero(T);
    if mA.uplo == 'U'
        @inbounds for jj in 1:n
            @fastmath s += vX[jj] * mA[jj, jj] * vY[jj]; #<! Diagonal
            @inbounds for ii in 1:(jj - 1)
                @fastmath s += vX[ii] * mA[ii, jj] * vY[jj] + vX[jj] * mA[ii, jj] * vY[ii];
            end
        end
    else #<! if A.uplo == 'L'
        @inbounds for jj in 1:n
            @fastmath s += vX[jj] * mA[jj, jj] * vY[jj]; #<! Diagonal
            @inbounds for ii in (jj + 1):n
                @fastmath s += vX[ii] * mA[ii, jj] * vY[jj] + vX[jj] * mA[ii, jj] * vY[ii];
            end
        end
    end

    return s;

end

# Based on https://discourse.julialang.org/t/125184 / https://stackoverflow.com/a/79385985 by Steven G. Johnson
function Adjugate( mA :: AbstractMatrix )
    
    ishermitian(mA) && return adjugate(Hermitian(mA));
    LinearAlgebra.checksquare(mA);
    sF = svd(mA);
    
    return sF.V * (Adjugate(Diagonal(sF.S)) * det(sF.Vt * sF.U)) * sF.U';

end

function Adjugate( mA :: LinearAlgebra.RealHermSymComplexHerm )
    
    sF = eigen(mA);
    
    return sF.vectors * Adjugate(Diagonal(sF.values)) * sF.vectors';

end

function Adjugate( mD :: Diagonal{<:Number} )

    vD = mD.diag;
    Base.require_one_based_indexing(vD);
    length(vD) < 2 && return Diagonal(one.(vD));

    # compute diagonal of adj(D): dadj[i] = prod(d) / d[i], but avoid dividing by zero
    dadj = similar(vD);
    prod = one(eltype(vD));
    for ii = 1:length(vD)
        dadj[ii] = prod;
        prod *= vD[ii];
    end
    prod = one(eltype(vD));
    for i = length(vD):-1:1
        dadj[i] *= prod;
        prod *= vD[i];
    end
    
    return Diagonal(dadj);

end

# using BenchmarkTools;
# using LinearAlgebra;
# using LoopVectorization;

# numRows = 10; numCols = 5; mA = randn(numRows, numCols); mB = Symmetric(randn(numRows, numRows)); mC = Matrix(mB); vX = randn(numRows); vY = randn(numCols);
# numRows = 100; numCols = 50; mA = randn(numRows, numCols); mB = Symmetric(randn(numRows, numRows)); mC = Matrix(mB); vX = randn(numRows); vY = randn(numCols);
# numRows = 1_000; numCols = 500; mA = randn(numRows, numCols); mB = Symmetric(randn(numRows, numRows)); mC = Matrix(mB); vX = randn(numRows); vY = randn(numCols);

# @btime dot($vX, $mA, $vY)
# @btime QuadForm($vX, $mA, $vY)
# @btime dot($vX, $mB, $vX)
# @btime QuadForm($vX, $mB, $vX) #<! Slow!
# @btime dot($vX, $mC, $vX)
# @btime QuadForm($vX, $mC, $vX) #<! Faster (No symmetry)
