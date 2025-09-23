# StackExchange Mathematics Q1385984
# https://math.stackexchange.com/questions/1385984
# Least Absolute Deviation (LAD) with Non Negative Constraint.
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
# - 1.0.000     23/09/2025  Royi Avital
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

function ObjFun( mA :: Matrix{T}, vX :: Vector{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    return sum(abs, mA * vX - vB);
    
end

function ProjSubGrad( mX :: Matrix{T}, hSubGrad :: Function, hProj :: Function; η :: T = 1.0 ) where {T <: AbstractFloat}

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        ηk = η / T(ii);
        
        mX[:, ii] .= hProj(mX[:, ii - 1] .- ηk .* hSubGrad(mX[:, ii - 1]));
    end

end

function ChamPock!( mX :: Matrix{T}, mK :: Matrix{T}, vB :: Vector{T}, vY :: Vector{T}, vX̄ :: Vector{T}, hProxF⁺ :: Function, hProxG :: Function, σ :: T, τ :: T; θ :: T = 1.0 ) where {T <: AbstractFloat}
    # Solving using Chambolle Pock algorithm (Also called Primal Dual Hybrid Gradient (PDHG) Method).
    # Solves: \arg \min_x f(K x) + g(x), f: Y ➡ [0, inf), g: X ➡ [0, inf).
    # Assumes efficient ProxF⁺ and ProxG.
    # Following the notations of Wikipedia.

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        vT = view(mX, :, ii - 1); #<! Previous iteration
        vX = view(mX, :, ii);
        
        # Calculation of `vY` depends on f() and should be adapted per function
        vY .= hProxF⁺(vY + σ * (mK * vX̄ - vB), σ);
        vX .= hProxG(vT - (τ * mK' * vY), τ);
        
        vX̄ .= vX + (θ * (vX - vT));
    end

end


## Parameters

# Data
numRows = 10;
numCols = 6;

# Solvers
numIterations = 25_000;

# Projected Sub Gradient
η = 1.05;

# PDHG 


## Load / Generate Data

mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

mX = zeros(numCols, numIterations);

dSolvers = Dict();

## Analysis

hObjFun(vX :: Vector{T}) where {T <: AbstractFloat} = ObjFun(mA, vX, vB);

# DCP Solver
vX0 = Variable(numCols);
sConvProb = minimize(Convex.norm_1(mA * vX0 - vB), vX0 >= 0.0);
solve!(sConvProb, ECOS.Optimizer; silent = true);

vXRef  = vX0.value
optVal = sConvProb.optval;

# Sub Gradient Method
methodName = "Sub Gradient";
hProj(vY :: Vector{T}) where {T <: AbstractFloat} = max.(vY, 0.0);
hSubGrad(vX :: Vector{T}) where {T <: AbstractFloat} = mA' * sign.(mA * vX - vB);

ProjSubGrad(mX, hSubGrad, hProj);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# Primal Dual Hybrid Gradient (PDHG) Method
# Solves: \arg \min f(Kx) + g(x) : || A x - b ||_1 + δᵪ(x) where δᵪ(x) is indicator for non negative values (Forces x ≥ 0)
# ProxF⁺: f⁺(p) <b, p> + I_{ || p ||_∞ ≤ 1 } -> ProxF⁺_{σ f⁺} (y) = Clip(y - σ b, -1, 1)
# ProxG: ProxG(y) = max(y, 0)
methodName = "PDHG";

hProxF⁺(vY :: Vector{T}, λ :: T) where {T <: AbstractFloat} = clamp.(vY, -1.0, 1.0);
hProxG(vY :: Vector{T}, λ :: T) where {T <: AbstractFloat} = max.(vY, 0.0);

mX .= 0.0;

# σ * τ * || A ||_2^2 ≤ 1
σ = 0.99 / opnorm(mA);
τ = σ;

vY = zeros(numRows);
vX̄ = zeros(numCols);

ChamPock!(mX, mA, vB, vY, vX̄, hProxF⁺, hProxG, σ, τ);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", line = attr(width = 3.0),
               text = methodName, name = methodName);
end
sLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

