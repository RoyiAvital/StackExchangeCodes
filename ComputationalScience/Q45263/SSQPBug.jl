## Packages

# Internal
using LinearAlgebra;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using OSQP;
using StableRNGs;
using StatusSwitchingQP;


## Constants & Configuration
RNG_SEED = 1234;

## Settings

oRng = StableRNG(1234);

## Functions

function SolveOSQP( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}; applyPolish :: Bool = true, dispReport :: Bool = false ) where {T <: AbstractFloat}

    numElements = length(vY);
    
    mP = sparse(I, numElements, numElements);
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

function SolveSSQP( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T} ) where {T <: AbstractFloat}

    numElements = length(vY);
    mI = collect(T.(I(numElements)));
    
    sQp = QP(mI; q = -vY, A = mA, b = vB, d = vL, u = vU);
    vX, _, _ = solveQP(sQp);

    return vX;

end

## Parameters

# Data
numRows = 50;
numCols = 75;

boxRadius = 10.0;


#%% Load / Generate Data

vY = randn(oRng, numCols);

mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

vL = -boxRadius .* rand(oRng, numCols);
vU = boxRadius .* rand(oRng, numCols);


## Analysis

# OSQP Solver
vX = SolveOSQP(vY, mA, vB, vL, vU);

# SSQP Solver
vX = SolveSSQP(vY, mA, vB, vL, vU);


## Run Time Analysis
@benchmark SolveOSQP($vY, $mA, $vB, $vL, $vU)
@benchmark SolveSSQP($vY, $mA, $vB, $vL, $vU)

## Display Results
