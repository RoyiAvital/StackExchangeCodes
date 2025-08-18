# StackExchange Cross Validated Q544135
# https://stats.stackexchange.com/questions/544135
# Projected Gradient Descent for Quadratic Programming Problem.
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
# - 1.0.000     18/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
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

function ObjFun( mX :: Matrix{T}, mA :: Matrix{T}, mB :: Matrix{T} ) where {T <: AbstractFloat}

    valObj = T(0.5) * sum(abs2, mA * mX - mB);

    return valObj;
    
end

function CVXSolver( mA :: Matrix{T}, mB :: Matrix{T} ) where {T <: AbstractFloat}

    numRows = size(mA, 2);
    numCols = size(mB, 2);
    mX = Convex.Variable(numRows, numCols);
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = minimize( T(0.5) * Convex.sumsquares(mA * mX - mB), mX >= zero(T) );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return mX.value;

end


## Parameters

# Data
numRowsA = 30;
numColsA = 27;

numRowsB = numRowsA;
numColsB = 18;

# Model

# Solver
numIterations = 1_000;
η = 5e-4;

## Load / Generate Data

mA = randn(oRng, numRowsA, numColsA);
mB = randn(oRng, numRowsB, numColsB);

hObjFun(mX :: Matrix{T}) where {T <: AbstractFloat} = ObjFun(mX, mA, mB);

dSolvers = Dict();

## Analysis
# Model: 0.5 * || A * X - B ||_2^2 s.t. X >= 0

# DCP Solver
methodName = "Convex.jl"

mXRef = CVXSolver(mA, mB);
optVal = hObjFun(mXRef);

dSolvers[methodName] = optVal * ones(numIterations);

# Projected Gradient Descent
methodName = "PGD";

mAA = mA' * mA;
mAB = mA' * mB;

hGradF(mX :: Matrix{T}) where {T <: AbstractFloat} = mAA * mX - mAB;
hProj(mY :: Matrix{T}) where {T <: AbstractFloat} = max.(mY, zero(T));

mXX = zeros(numColsA, numColsB);
mX  = zeros(numColsA, numColsB, numIterations);

for ii in 2:numIterations
    global mXX;
    mXX = hProj(mXX .- η * hGradF(mXX));

    mX[:, :, ii] = mXX;
end

dSolvers[methodName] = [hObjFun(mX[:, :, ii]) for ii ∈ 1:size(mX, 3)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end