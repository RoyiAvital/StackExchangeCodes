# StackOverflow Q64422417
# https://stackoverflow.com/questions/64422417
# IRLS vs. Linear Programming for Large Scale for Least Absolute Deviation (LAD) Regression.
# References:
#   1.  Mathematics Q2603548 - Least Absolute Deviation (LAD) Line Fitting / Regression.
#   2.  Signal Processing Q91788 - Fit Data Samples with a Robust Fit.
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     30/06/2024  Royi Avital
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
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function RandLaplace( oRng :: AbstractRNG, dataType :: Type{T}; μ :: T = T(0.0), β :: T = T(1.0) ) where {T <: AbstractFloat}
    # https://math.stackexchange.com/questions/1588871

    val1 = rand(oRng, dataType);
    val2 = rand(oRng, dataType);

    x = log(val1 / val2); #<! Laplace with μ = 0, β = 1

    return (β * x) + μ;

end

function RandLaplace( dataType :: Type{T}; μ :: T = T(0.0), β :: T = T(1.0) ) where {T <: AbstractFloat}

    return RandLaplace(Random.default_rng(), dataType; μ = μ, β = β);

end

function RandLaplace( oRng :: AbstractRNG; μ :: T = 0.0, β :: T = 1.0 ) where {T <: AbstractFloat}

    return RandLaplace(oRng, Float64;  μ = μ, β = β);

end

function RandLaplace( ; μ :: T = 0.0, β :: T = 1.0 ) where {T <: AbstractFloat}

    return RandLaplace(Random.default_rng(), Float64; μ = μ, β = β);

end

function CVXSolver( mA :: AbstractMatrix{T}, vY :: AbstractVector{T}, modelNorm :: T )  where {T <: AbstractFloat}
    
    numCols = size(mA, 2);
    
    vX = Variable(numCols);
    sConvProb = minimize(norm(mA * vX - vY, modelNorm));
    solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);

end


## Parameters

# Problem parameters
numRows = 100_000; #<! Matrix A
numCols = 15;  #<! Matrix A
μ = 0.0;
β = 0.05;

# Model
modelNorm = 1.0; #<! 1 <= modelNorm <= inf

# Solver Parameters
numIterations   = Unsigned(25);


#%% Load / Generate Data

mA = randn(oRng, numRows, numCols);
vXᶲ = randn(oRng, numCols); #<! Reference
vN = [RandLaplace(oRng; μ = μ, β = β) for _ ∈ 1:numRows];
vY = mA * vXᶲ + vN;


hObjFun( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = norm(mA * vX - vY, modelNorm);

dSolvers = Dict();


## Analysis
# The Model: \arg \minₓ || A x - y ||₁

# DCP Solver
methodName = "Convex.jl"

vXRef = CVXSolver(mA, vY, modelNorm);

dSolvers[methodName] = hObjFun(vXRef) * ones(numIterations);

# Iterative Reweighted Least Squares (IRLS)
methodName = "IRLS";

mX = zeros(numCols, numIterations);
vT  = zeros(numCols);
vW  = zeros(numRows);
mWA = zeros(size(mA));
mC  = zeros(numCols, numCols);
sBkWorkSpace = BunchKaufmanWs(mC);

for ii = 2:numIterations
    vZ = mX[:, ii - 1];
    vZ = IRLS!(vZ, mA, vY, vW, mWA, mC, vT, sBkWorkSpace; normP = modelNorm, numItr = UInt32(1));
    mX[:, ii] .= vZ;
end

# vX = IRLS(mA, vY; normP = modelNorm, numItr = numIterations);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Run Time Analysis
runTime = @belapsed CVXSolver(mA, vY, modelNorm) seconds = 2;
resAnalysis = @sprintf("The Convex.jl (ECOS) solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

runTime = @belapsed IRLS(mA, vY; normP = modelNorm, numItr = numIterations) seconds = 2;
resAnalysis = @sprintf("The IRLS solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);