# StackExchange Mathematics Q3620445
# https://math.stackexchange.com/questions/3620445
# The Sub Gradient of the Spectral Norm (Schatten ∞ Norm).
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     26/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( vX :: Vector{T}, tA :: Array{T, 3}, mB :: Matrix{T} ) where {T <: AbstractFloat}

    mAA = zero(mB);
    for ii in 1:length(vX);
        mAA .+= vX[ii] .* tA[:, :, ii];
    end

    mAA .-= mB;

    return opnorm(mAA);
    
end

function ∇ObjFun( vX :: Vector{T}, tA :: Array{T, 3}, mB :: Matrix{T} ) where {T <: AbstractFloat}
    
    numRows, numCols, numMat = size(tA);
    
    mÂ = reshape(tA, numRows * numCols, numMat);
    mC = reshape(mÂ * vX, numRows, numCols) - mB;
    sF = svd(mC);

    return mÂ' * vec(sF.U[:, 1] * sF.V[:, 1]');

end


## Parameters

# Data
numRows = 4;
numCols = 3;
numMat  = 2;


## Load / Generate Data

tA = randn(numRows, numCols, numMat);
mB = randn(numRows, numCols);

vX = randn(numMat);

h∇ObjFun( vX :: Vector{T} ) where {T <: AbstractFloat} = ∇ObjFun(vX, tA, mB);
hObjFun( vX :: Vector{T} ) where {T <: AbstractFloat} = ObjFun(vX, tA, mB);

## Analysis

CalcFunGrad(vX, hObjFun) - h∇ObjFun(vX)

## Display Results

