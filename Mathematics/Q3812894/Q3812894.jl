# StackExchange Mathematics Q1586207
# https://math.stackexchange.com/questions/1586207
# Solve Linear Least Squares with Box Constraints (Bounded Least Squares).
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
# - 1.0.000     20/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;                #<! Usually more accurate than SCS
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCS;                 #<! Seems to support more cases for Continuous optimization than ECOS
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function CVXSolver( mA :: Matrix{T}, vB :: Vector{T}, vL :: Vector{T}, vU :: Vector{T} ) where {T <: AbstractFloat}

    numCols = size(mA, 2);

    vX = Variable(numCols);
    
    sConvProb = minimize( Convex.sumsquares(mA * vX - vB), [vX <= vU, vX >= vL] ); #<! https://github.com/jump-dev/Convex.jl/issues/722
    # Convex.solve!(sConvProb, SCS.Optimizer; silent = true);
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);

end


## Parameters

# Data
numRows = 2_000;
numCols = 500;


## Load / Generate Data

mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

vL = zeros(numCols);
vU = ones(numCols);

hObjFun( vX :: Vector{T} ) where {T <: AbstractFloat} = 0.5 * sum(abs2, mA * vX - vB);

dSolvers = Dict();


## Analysis
# The Model: \arg \minₓ || A x - b ||² subject to x ≤ u, x ≥ l

# DCP Solver
methodName = "Convex.jl"

vXRef = CVXSolver(mA, vB, vL, vU);

# dSolvers[methodName] = hObjFun(vXRef) * ones(numIterations);
# optVal = hObjFun(vXRef);

# KKT Conditions (QR Boxed \ Projected Newton Method)
methodName = "Projected Newton Method";

vX, isConverged = LsqBox(mA, vB, vL, vU);

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

# figureIdx += 1;

# sTr1 = scatter(; x = vλ, y = vV, mode = "lines", 
#               line_width = 2.75,
#               name = "ϕ(λ)", text = "ϕ(λ)");
# sTr2 = scatter(; x = [λꜛ], y = [hϕ(λꜛ)], mode = "scatter", 
#               marker_size = 7,
#               name = "λꜛ (Binary Search)", text = "λꜛ");
# sTr3 = scatter(; x = vλ, y = repeat([0.0], numGridPts), mode = "lines", 
#               line_width = 1.25,
#               name = "Optimal Value", text = "Optimal Value");
# sTr4 = scatter(; x = vλ, y = repeat([-valA], numGridPts), mode = "lines", 
#               line_width = 1.25,
#               name = "-a", text = "-a");
# sLayout = Layout(title = "ϕ(λ)", width = 600, height = 600, 
#                  xaxis_title = "λ", yaxis_title = "ϕ(λ)",
#                  yaxis_range = (-1.0, 1.0),
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
#                  legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = Plot([sTr1, sTr2, sTr3, sTr4], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end

# Run Time Analysis
runTime = @belapsed CVXSolver(mA, vB, vL, vU) seconds = 2;
resAnalysis = @sprintf("The Convex.jl (ECOS) solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

runTime = @belapsed LsqBox(mA, vB, vL, vU) seconds = 2;
resAnalysis = @sprintf("The Projected Newton Method solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);



