# StackExchange Mathematics Q5087076
# https://math.stackexchange.com/questions/5087190
# Solve L1 Regularization where $\left\| \boldsymbol{A} \boldsymbol{x} \right\| \ll \left\| \boldsymbol{x} \right\|$.
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
# - 1.0.000     05/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
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

function CVXSolver( mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    numCols = size(mA, 2);

    vX = Convex.Variable(numCols);
    sConvProb = minimize( Convex.norm(mA * vX - vB, 1) );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);
    
end


## Parameters

# Data
fileUrl = "";
fileName = "Data.csv";

numRows = 11;
numCols = 9;

normP = 1.0;


## Load / Generate Data

vData = readdlm(fileName, ',', Float64);
# vData = readdlm(download(fileUrl), ',', Float64);

mA = collect(reshape(vData[1:(numRows * numCols)], numCols, numRows)');
vB = vData[(end - numRows + 1):end];


## Analysis

vXRef = CVXSolver(mA, vB);
vX = IRLS(mA, vB, normP; numItr = 5_000, ϵ = 1e-10);

maximum(abs.(vX - vXRef))


## Display Results

