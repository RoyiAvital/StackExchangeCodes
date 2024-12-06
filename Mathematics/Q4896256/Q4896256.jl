# StackExchange Mathematics Q4896256
# https://math.stackexchange.com/questions/4896256
# Solve a Quadratic Form with a Unit Simplex Constraint.
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
# - 1.0.000     04/12/2024  Royi Avital
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
using SparseArrays;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function f( vX :: Vector{T}, mA :: Matrix{T} ) where {T <: AbstractFloat}

    return dot(vX, mA, vX);
    
end

function ∇f( vX :: Vector{T}, mA :: Matrix{T} ) where {T <: AbstractFloat}
    # Does not assume `mA` is symmetric

    return (mA * vX) + (mA' * vX);

end


## Parameters

# Data
numElements = 2; #<! Number of elements
forcePsd    = false;
ϵ           = 1e-5;

# Solver
numIter = 1_000;
η       = 1e-3;

# Visualization
numGridPts = 500;


## Load / Generate Data

mA = randn(oRng, numElements, numElements); #<! Each slice on 3rd dimension is `mAi`

if forcePsd
    mA = mA' * mA;
    mA = 0.5 * (mA' * mA);
end


## Analysis

hF( vX :: Vector{T} ) where {T <: AbstractFloat} = f(vX, mA);
h∇f( vX :: Vector{T} ) where {T <: AbstractFloat} = ∇f(vX, mA);
hProjFun( vX :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = ProjSimplexBall(vX); 

# Validate Gradient
vX = randn(oRng, numElements);
@assert (maximum(abs.(h∇f(vX) - CalcFunGrad(vX, hF))) <= ϵ) "The gradient calculation is not verified";

# Validate Projection
ballRadius = 1.35;
vY = randn(21);
vB = ProjSimplexBall(vY; ballRadius = ballRadius);

vC = Variable(length(vY));
sConvProb = minimize( Convex.sumsquares(vC - vY), [vC >= 0.0, sum(vC) == ballRadius] );
solve!(sConvProb, ECOS.Optimizer; silent = true);
vC = vC.value[:];
optVal = sConvProb.optval;

@assert (abs(sum(vB) - ballRadius) <= ϵ) "The projection calculation is not verified";
@assert (-minimum(vB) <= ϵ) "The projection calculation is not verified";
@assert (abs(sum(abs2, vB - vY) - sum(abs2, vC - vY)) <= ϵ) "The projection calculation is not verified";

# Solve

vZ = ProximalGradientDescentAcc(hProjFun(zeros(numElements), 0.0), h∇f, hProjFun, η, numIter);

# Path
mX = zeros(numElements, numIter);
mX[:, 1] = hProjFun(mX[:, 1], 0.0); #<! Make it feasible
for ii ∈ 2:numIter
    mX[:, ii] = hProjFun(mX[:, ii - 1] - η * h∇f(mX[:, ii - 1]), 0.0);
end


## Display Analysis

if (numElements == 2)
figureIdx += 1;
vG = LinRange(-0.1, 1.1, numGridPts);
mO = zeros(numGridPts, numGridPts);
vT = zeros(2);

for jj ∈ 1:numGridPts, ii ∈ 1:numGridPts
    # x, y notation (Matches `heatmap()`)
    vT[1] = vG[jj];
    vT[2] = vG[ii];
    mO[ii, jj] = hF(vT);
end

if forcePsd
    oTr1 = heatmap(; x = vG, y = vG, z = log1p.(mO));
else
    oTr1 = heatmap(; x = vG, y = vG, z = mO);
end
oTr2 = scatter(; x = mX[1, :], y = mX[2, :], mode = "markers", name = "Path");
oLayout = Layout(title = "Objective Function - Log Scale", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

end

