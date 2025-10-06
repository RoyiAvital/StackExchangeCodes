# StackExchange Cross Validated Q632389
# https://stats.stackexchange.com/questions/632389
# Least Squares with 2 Terms and Unit Simplex Constraints.
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
# - 1.0.000     06/10/2025  Royi Avital
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

function ObjFun( vW :: Vector{T}, mC :: Matrix{T} ) where {T <: AbstractFloat}

    valObj = T(0.5) * sum(abs2, mC * vW);

    return valObj;
    
end

function CVXSolver( mC :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim = size(mC, 2) ÷ 2;

    vU = Convex.Variable(dataDim);
    vV = Convex.Variable(dataDim);
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = minimize( T(0.5) * Convex.sumsquares(mC * [vU; vV]), [vU >= zero(T), Convex.sum(vU) == 1, vV >= zero(T), Convex.sum(vV) == 1] );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    vW = [vec(vU.value); vec(vV.value)];
    
    return vW;

end


## Parameters

# Data
numRows = 10;
dataDim = 4; #<! Length of `vU` / `vV`
numCols = 2 * dataDim; #<! Length of `vW` (Must be even)

# Solver
numIterations = 1_000;
η = 5e-4;

## Load / Generate Data

mC = randn(oRng, numRows, numCols);

hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vW, mC);

dSolvers = Dict();

## Analysis
# Model: 0.5 * || C * w ||_2^2 s.t. u ∈ Δⁿ, v ∈ Δⁿ
# where w = [uᵀ, vᵀ]ᵀ

# DCP Solver
methodName = "Convex.jl"

vWRef = CVXSolver(mC);
optVal = hObjFun(vWRef);

dSolvers[methodName] = optVal * ones(numIterations);

# Projected Gradient Descent
methodName = "PGD";

mCC = mC' * mC;

hGradF(vW :: Vector{T}) where {T <: AbstractFloat} = mCC * vW;
hProj(vW :: Vector{T}) where {T <: AbstractFloat} = [ProjectSimplexBall(vW[1:dataDim]); ProjectSimplexBall(vW[(dataDim + 1):end])];

mX = zeros(numCols, numIterations);

for ii in 2:numIterations
    mX[:, ii] = hProj(mX[:, ii - 1] .- η * hGradF(mX[:, ii - 1]));
end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# Projected Gradient Descent
methodName = "Accelerated PGD";

mX  = zeros(numCols, numIterations);
vZ  = zeros(numCols);

for ii in 2:numIterations
    # FISTA (Nesterov) Accelerated
    mX[:, ii] = hProj(vZ .- η * hGradF(vZ));
    fistaStepSize = (ii - 1) / (ii + 2);
    vZ .= mX[:, ii] .+ (fistaStepSize .* (mX[:, ii] .- mX[:, ii - 1]));
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