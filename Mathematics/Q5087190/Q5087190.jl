# StackExchange Mathematics Q5087190
# https://math.stackexchange.com/questions/5087190
# The Gradient of an LDA (Linear Discriminant Analysis) Like Objective Function.
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
# - 1.0.000     02/08/2025  Royi Avital
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
include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( mW :: Matrix{T}, mS :: Matrix{T}, mP :: Matrix{T} ) where {T <: AbstractFloat}

    valNum = det(mW' * mS * mW);
    valDen = det(mW' * mP * mW);

    return valNum / valDen;
    
end

function ∇ObjFun( mW :: Matrix{T}, mS :: Matrix{T}, mP :: Matrix{T} ) where {T <: AbstractFloat}

    mWSW  = mW' * mS * mW;
    mWPW  = mW' * mP * mW;
    mWStW = mW' * mS' * mW;
    mWPtW = mW' * mP' * mW;
    
    detS  = det(mWSW);
    detP  = det(mWPW);
    detSt = det(mWStW);
    detPt = det(mWPtW);

    mG = ((mS * mW * Adjugate(mWSW)) / detP) + ((mS' * mW * Adjugate(mWStW)) / detPt) - ((detS / (detP * detP)) * mP * mW * Adjugate(mWPW)) - ((detSt / (detPt * detPt)) * mP' * mW * Adjugate(mWPtW));


    # By Greg's answer (https://math.stackexchange.com/a/5087385)
    # mM = mWSW;
    # mN = mWPW;
    # α = detS;
    # β = detP;
    # valJ = α / β;

    # mG = valJ * (mS' * mW / mM' + mS * mW / mM - mP' * mW / mN' - mP * mW / mN);

    return mG;
    
end


## Parameters

# Data
numRows = 10;
numCols = 6;


## Load / Generate Data

mW = randn(numRows, numCols);
mS = randn(numRows, numRows);
mP = randn(numRows, numRows);

## Analysis

hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = ObjFun(reshape(vW, numRows, numCols), mS, mP);

# Finite Differences solution
mGRef = reshape(CalcFunGrad(mW[:], hObjFun; ε = 1e-6), numRows, numCols);
mG = ∇ObjFun(mW, mS, mP);

maximum(abs.(mG - mGRef))


## Display Results

