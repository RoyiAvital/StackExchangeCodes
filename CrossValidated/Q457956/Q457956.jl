# StackExchange Cross Validated Q457956
# https://stats.stackexchange.com/questions/457956
# Solve LASSO Like Problem with 2 L1 Terms.
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

function ObjFun( vX :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, λ :: T, μ :: T, valT :: T ) where {T <: AbstractFloat}

    valObj = T(0.5) * sum(abs2, mA * vX - vB) + λ * sum(abs, vX) + μ * sum(abs, vX .- valT);

    return valObj;
    
end

function CVXSolver( mA :: Matrix{T}, vB :: Vector{T}, λ :: T, μ :: T, valT :: T ) where {T <: AbstractFloat}

    numCols = size(mA, 2);
    vX = Convex.Variable(numCols);
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = minimize( T(0.5) * Convex.sumsquares(mA * vX - vB) + λ * Convex.norm(vX, 1) + μ * Convex.norm(vX - valT, 1) );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);

end


## Parameters

# Data
numRows = 10;
numCols = 6;

# Model
λ = 0.75;
μ = 0.65;
valT = 0.15;

# Solver
numIterations = 100;
γ = 0.05;

## Load / Generate Data

mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

hObjFun(vX :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vX, mA, vB, λ, μ, valT);

dSolvers = Dict();

## Analysis
# Model: 0.5 * || A * x - b ||_2^2 + λ || x ||_1 + μ * || x - t ||_1

# DCP Solver
methodName = "Convex.jl"

vXRef = CVXSolver(mA, vB, λ, μ, valT);
optVal = hObjFun(vXRef);

dSolvers[methodName] = optVal * ones(numIterations);

# 3 Operators Split
methodName = "3 Operators";

mAA = mA' * mA;
vAb = mA' * vB;

hGradF(vX :: Vector{T}) where {T <: AbstractFloat} = mAA * vX - vAb;
hProxG(vY :: Vector{T}, λ :: T) where {T <: AbstractFloat} = sign.(vY) .* max.(zero(T), abs.(vY) .- λ);
hProxH(vY :: Vector{T}, λ :: T) where {T <: AbstractFloat} = valT .+ hProxG(vY .- valT, λ);

mX = zeros(numCols, numIterations);
vZ = zeros(numCols);

for ii in 2:numIterations
    vXg = hProxG(vZ, γ * λ);
    vXh = hProxH(2.0 * vXg - vZ - γ * hGradF(vXg), γ * μ);
    vZ .+= vXh .- vXg;

    mX[:, ii] = vXg;
end

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