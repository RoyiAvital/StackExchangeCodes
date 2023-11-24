# StackExchange Mathematics Q2230545
# https://math.stackexchange.com/questions/2230545
# Gradient Descend for Quadratic Function with Non Negative Constraints with Constant Step Size.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
#   3. 
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     24/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
using MAT;
using PlotlyJS;
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions

hObjFun( vU :: Vector{T}, vA :: Vector{T}, vX :: Vector{T}, mB :: Matrix{T}, γ :: T, λ :: T ) where {T <: AbstractFloat} = 0.5 * sum(abs2, vU - vA) + 0.5γ * sum(abs2, mB' * vU - vX) + λ * sum(vU);
∇ObjFun( vU :: Vector{T}, vA :: Vector{T}, vX :: Vector{T}, mB :: Matrix{T}, γ :: T, λ :: T ) where {T <: AbstractFloat} = vU - vA + γ * mB * (mB' * vU - vX) .+ λ;


## Parameters

# Data
numRows = 500;
numCols = 200;
γ       = 1.2;
λ       = 0.5;

# Solvers
numIterations = 25000;
hΗ( mB :: Matrix{T}, γ :: T ) where {T <: AbstractFloat} = (1 / (1 + γ * (opnorm(mB) ^ 2))) - (length(mB) * eps(one(T)));

## Generate / Load Data
oRng = StableRNG(1234);
mB = randn(oRng, numRows, numCols);
vA = randn(oRng, numRows);
vX = randn(oRng, numCols);

vU = zeros(numRows);

hF( vU :: Vector{T} ) where {T <: AbstractFloat} = hObjFun(vU, vA, vX, mB, γ, λ);
∇f( vU :: Vector{T} ) where {T <: AbstractFloat} = ∇ObjFun(vU, vA, vX, mB, γ, λ);
hProj( vU :: Vector{T} ) where {T <: AbstractFloat} = max.(vU, zero(T));

dSolvers = Dict();

## Analysis

# DCP Solver
vU0 = Variable(numRows);
sConvProb = minimize(0.5 * sumsquares(vU0 - vA) + 0.5γ * sumsquares(mB' * vU0 - vX) + λ * sum(vU0), vU0 >= 0);
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
vURef = copy(vU0.value);
optVal = sConvProb.optval;


# Projected Gradient Descent
methodName = "Projected Gradient Descent";
η = hΗ(mB, γ);

mU = zeros(numRows, numIterations);
mU[:, 1] .= vU;

for ii in 2:numIterations
    mU[:, ii] = hProj(mU[:, ii - 1] .- η * ∇f(mU[:, ii - 1]));
end



dSolvers[methodName] = [hF(mU[:, ii]) for ii ∈ 1:size(mU, 2)];

## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function, Condition Number = $(cond(mB))", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
