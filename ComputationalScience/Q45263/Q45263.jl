# StackExchange Computational Science Q45263
# https://scicomp.stackexchange.com/questions/45263
# Solve Projection Problem with Linear Equality and Box Cosntraints.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     25/05/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using Convex;
using ECOS;
# using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using OSQP;
using StableRNGs;
# using StatusSwitchingQP;


## Constants & Configuration
RNG_SEED = 1234;

# juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
# include(joinpath(juliaCodePath, "JuliaInit.jl"));
# include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function SolveCVX( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T} ) where {T <: AbstractFloat}

    numElements = length(vY);

    vX = Convex.Variable(numElements);

    sConvProb = minimize( Convex.sumsquares(vX - vY), [mA * vX == vB, vX >= vL, vX <= vU] ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);

end

function SolveOSQP( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}; applyPolish :: Bool = true, dispReport :: Bool = false ) where {T <: AbstractFloat}

    numElements = length(vY);
    
    mP = sparse(T, I, numElements, numElements);
    vQ = -vY;

    mAA = sparse(mA);
    mAA = vcat(mAA, sparse(I, numElements, numElements));
    vLL = vcat(vB, vL);
    vUU = vcat(vB, vU);

    sOSQP = OSQP.Model();
    OSQP.setup!(sOSQP; P = mP, q = vQ, A = mAA, l = vLL, u = vUU, polish = applyPolish, verbose = dispReport); #<! Polish allows high accuracy solution
    sRes = OSQP.solve!(sOSQP);

    return sRes.x;

end

function SolveDysktra( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}; numIterations = 50 ) where {T <: AbstractFloat}

    numElements = length(vY);
    vX = copy(vY);
    vZ = zeros(T, numElements);
    vP = zeros(T, numElements);
    vQ = zeros(T, numElements);
    vT = zeros(T, numElements);

    sSvd = svd(mA);
    mVV = sSvd.V * sSvd.Vt;
    # mVS⁺Uᵗ = sSvd.V * Diagonal(inv.(sSvd.S)) * sSvd.U';

    vBB = sSvd.V * Diagonal(inv.(sSvd.S)) * sSvd.U' * vB;
    
    for _ in 1:numIterations

        vZ .= vX .+ vP;
        # Project `vZ` onto the Linear Equality
        mul!(vT, mVV, vZ);
        # vZ .= vZ .- vT .+ vBB;
        vZ .+= vBB .- vT;

        vP .+= vX .- vZ;

        vX .= vZ .+ vQ;
        # Project `vX` onto the Box Constraints
        vX .= clamp.(vX, vL, vU);
        vQ .+= vZ .- vX;

    end

    return vX;

end

# function SolveSSQP( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T} ) where {T <: AbstractFloat}

#     numElements = length(vY);
#     mI = collect(T.(I(numElements)));
    
#     sQp = QP(mI; q = -vY, A = mA, b = vB, d = vL, u = vU);
#     vX, _, _ = solveQP(sQp);

#     return vX;

# end

## Parameters

# Data
numRows = 50;
numCols = 75;

boxRadius = 10.0;

# Solver
maxRadius = 0.5;


#%% Load / Generate Data

vY = randn(oRng, numCols);

mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

vL = -boxRadius .* rand(oRng, numCols);
vU = boxRadius .* rand(oRng, numCols);


## Analysis

# DCP Solver
vXRef = SolveCVX(vY, mA, vB, vL, vU);

# OSQP Solver
vX = SolveOSQP(vY, mA, vB, vL, vU);

# Dysktra Solver
vX = SolveDysktra(vY, mA, vB, vL, vU);


## Run Time Analysis
@benchmark SolveCVX($vY, $mA, $vB, $vL, $vU)
@benchmark SolveOSQP($vY, $mA, $vB, $vL, $vU)
@benchmark SolveDysktra($vY, $mA, $vB, $vL, $vU; numIterations = 75)

## Display Results

f(vX) = mA' * ((mA * mA') \ (mA * vX));

sQr = qr(mA);
g(vX) = sQr.R' * ((sQr.R * sQr.R') \ (sQr.R * vX));

sSvd = svd(mA);
g(vX) = sSvd.V * sSvd.Vt * vX;


f(vX) = mA' * ((mA * mA') \ vX);

sSvd = svd(mA);
g(vX) = sSvd.V * Diagonal(inv.(sSvd.S)) * sSvd.U' * vX;


