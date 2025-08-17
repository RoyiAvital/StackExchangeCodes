# StackExchange Mathematics Q264099
# https://math.stackexchange.com/questions/264099
# Solving the Primal Kernel SVM Problem.
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
# - 1.0.000     12/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function KernelSVM( vα :: Vector{T}, paramB :: T, mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}

    numSamples = length(vY);
    
    # Regularization
    objVal = 0.5 * λ * dot(vα, mK, vα);

    # Objective
    vH = [max(zero(T), T(1) - vY[ii] * (dot(vα, mK[:, ii]) + paramB)) for ii in 1:numSamples];
    if squareHinge
        vH = [vH[ii] * vH[ii] for ii in 1:numSamples];
    end

    objVal += sum(vH);

    # Vectorized form
    # objVal = 0.5 * λ * dot(vα, mK, vα) + sum(max.(zero(T), T(1) .- vY .* (mK * vα .+ paramB)));

    return objVal;

end


## Parameters

# Data
numRows = 10;

# Solver
α  = 1.0;
vα = collect(LinRange(0.1, 1.5, 10));

# Visualization


## Load / Generate Data


## Analysis

mA = randn(numRows, numRows);
mA = mA' * mA;
mA = 0.5 * (mA' + mA);
mC = (mC' * mC) + 0.05 * I;
mC = 0.5 * (mC' + mC); #<! SPD Matrix

vB = randn(numRows);

# Pre Process
mU = cholesky(mX).U;
vY = mU * vX;
vD = mU' \ vB;
mE = mU' \ mA / mU; #<! Symmetric (Can be SPD / SPSD as `mA`)
mH = hessenberg(mE);

## Display Analysis

figureIdx += 1;

# startX, endX = CalcXRangeLine(vW, tuRange);
# vXX = collect(LinRange(startX, endX, numGridPts));
# vYY = CalcLineVal(vW, vXX);

# sTr1 = scatter(x = mX[:, 1], y = mX[:, 2], mode = "markers", text = "Data Samples", name = "Data Samples",
#                 marker = attr(size = 10, color = vY));
# sTr2 = scatter(x = vXX, y = vYY, 
#                 mode = "lines", text = "Classifier", name = "Classifier",
#                 line = attr(width = 2.5));
# sTr3 = scatter(x = mX[vSuppVecIdx, 1], y = mX[vSuppVecIdx, 2], mode = "markers", text = "Support Vectors", name = "Support Vectors",
#                 marker = attr(size = 20, color = "rgba(0, 0, 0, 0)", line = attr(color = "k", width = 2)));
# sLayout = Layout(title = "The Data Samples and Classifier", width = 600, height = 600, hovermode = "closest",
#                 xaxis_title = "t", yaxis_title = "y", xaxis_range = tuRange, yaxis_range = tuRange,
#                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = plot([sTr1, sTr2, sTr3], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end

