# StackOverflow Q49787068
# https://stackoverflow.com/questions/49787068
# Minimize L1 Norm with Matrix Linear Equality Constraint.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   3. 
# TODO:
# 	1.  C
# Release Notes
# - 1.1.000     26/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added Accelerated Proximal Gradient Descent.
# - 1.0.000     26/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
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

function ProjLinEqualitySet!( vY :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    # Can be optimized by using the factorization of `mA`
    vY .-= mA' * ((mA * mA') \ (mA * vY - vB));
    
end


## Parameters

# Data
numRows = 20;
numCols = 50; #<! Symmetric Matrix
δ       = 1e6;
δTol    = 1e-5;

# Solvers
numIterations = 10_000;

# Projected Gradient Descent
η = 0.00005;

## Generate / Load Data
oRng = StableRNG(1234);
mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

# See https://discourse.julialang.org/t/73206
hδFun( vX :: Vector{<: AbstractFloat} ) = δ * !isapprox(mA * vX, vB; atol = δTol);
hObjFun( vX :: Vector{<: AbstractFloat} ) = sum(abs, vX) + hδFun(vX);

dSolvers = Dict();


## Analysis

# DCP Solver
vX0 = Variable(numCols);
# Since mX0 and mB are SPSD the `tr()` in non negative.
# Hence one could use `abs()` to avoid complex numbers.
sConvProb = minimize(norm(vX0, 1), mA * vX0 == vB);
solve!(sConvProb, SCS.Optimizer; silent = true);
vXRef = vX0.value
optVal = sConvProb.optval;

# Projected Gradient Descent (Proximal Gradient Descent)
methodName = "PGD";

∇F( vY :: Vector{T} ) where {T <: AbstractFloat} = sign.(vY);
hProjFun( vY :: Vector{T} ) where {T <: AbstractFloat} = ProjLinEqualitySet!(vY, mA, vB);

mX = zeros(numCols, numIterations);
mX[:, 1] .= mA \ vB;

for ii ∈ 2:numIterations
    mX[:, ii] = mX[:, ii - 1] .- η * ∇F(mX[:, ii - 1]); #<! Gradient step
    mX[:, ii] = hProjFun(mX[:, ii]); #<! Projection step
end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# Accelerated Projected Gradient Descent (Accelerated Proximal Gradient Descent)
methodName = "Acc PGD";

mX = zeros(numCols, numIterations);
mX[:, 1] .= mA \ vB;
vZ = copy(mX[:, 1]);
∇vZ = copy(mX[:, 1]);

for ii ∈ 2:numIterations
    ω = ii / (ii + 3);
    # ω = (ii - 1) / (ii + 2);

    ∇vZ .= ∇F(vZ);
    mX[:, ii] .= vZ .- (η .* ∇vZ);
    mX[:, ii] = hProjFun(mX[:, ii]); #<! Projection step

    vZ .= mX[:, ii] .+ (ω .* (mX[:, ii] .- mX[:, ii - 1]))
end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function, Condition Number = $(@sprintf("%0.3f", cond(mA)))", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end