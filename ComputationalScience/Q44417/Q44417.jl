# StackExchange Computational Science Q44417
# https://scicomp.stackexchange.com/questions/44417
# Efficient Solver for Solving a Large Linear System Sequentially of a Positive Definite Matrix.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     31/07/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions


## Parameters

# Problem parameters
numRows = 5_000; #<! Matrix K
numCols = numRows;  #<! Matrix K



#%% Load / Generate Data

mA = randn(oRng, numRows, numCols);
mA = (mA' * mA) + I; #<! PD 
mB = randn(oRng, numRows, numCols);


## Analysis

# Pre Work
oChol  = cholesky(mA; check = false);

## Display Results

runTime = @belapsed ($oChol \ $mB) seconds = 2;
resAnalysis = @sprintf("Direct Solver based on Cholesky run time: %0.5f [Sec]", runTime);
println(resAnalysis);

# The CG solver will require few iterations per solution and 2 mutliplcations per iteration.
# Neglecting all the other calculations per iteration.
runTime = @belapsed ($mA * $mB) seconds = 2;
resAnalysis = @sprintf("Iterative Solver based on Matrix Multiplication run time: %0.5f [Sec]", runTime);
println(resAnalysis);


