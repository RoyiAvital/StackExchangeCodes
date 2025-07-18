# StackExchange Mathematics Q3832410
# https://math.stackexchange.com/questions/3832410
# Efficient Solver for $\arg \min_{\boldsymbol{x}} \sum_{i = 1}^{n} {\left\| \boldsymbol{a}_{i} - \boldsymbol{x} \right\|}_{\infty}^{2}$ with Large $n$.
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
# - 1.0.000     14/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function SolveProblemConvex( mA :: Matrix{T}; useSquare :: Bool = true ) where {T <: AbstractFloat}

    dataDim    = size(mA, 1);
    numSamples = size(mA, 2);

    vX = Variable(dataDim);

    vV        = vcat([Convex.norm_inf(vX - mA[:, ii]) for ii ∈ 1:numSamples]...);
    if useSquare
        sConvProb = minimize( Convex.sumsquares(vV) ); #<! See (https://github.com/jump-dev/Convex.jl/issues/722)
    else
        sConvProb = minimize( Convex.sum(vV) );
    end
    
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);
    
end

function CostFun( vX :: Vector{T}, mA :: Matrix{T} ) where {T <: AbstractFloat}
    
    numSamples = size(mA, 2);
    
    valCost = zero(T);

    for ii ∈ 1:numSamples
        valCost += norm(vX - mA[:, ii], Inf);
    end

    return valCost;
    
end


## Parameters

# Data
dataDim    = 7;
numSamples = 100;


## Load / Generate Data

mA = randn(oRng, dataDim, numSamples);


## Analysis

vX = SolveProblemConvex(mA);
mE = extrema(mA; dims = 2);

vXX = zeros(dataDim);
for ii ∈ 1:dataDim
    vXX[ii] = mean(mE[ii]);
end

# Ideas for simple solvers
valCostOpt    = CostFun(vX, mA);
valCostEst    = CostFun(vXX, mA);
valCostMean   = CostFun(vec(mean(mA; dims = 2)), mA);
valCostMedian = CostFun(vec(median(mA; dims = 2)), mA);

println("Cost optimal value: $(valCostOpt)");
println("Cost approximation value: $(valCostEst)");
println("Cost mean value: $(valCostMean)");
println("Cost median value: $(valCostMedian)");


## Display Analysis


