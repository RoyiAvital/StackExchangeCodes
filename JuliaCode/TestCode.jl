# Test Code
# Several test for the Julia Code.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     23/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using BenchmarkTools;
using UnicodePlots;

## Constants & Configuration

## External
include("JuliaInit.jl");
include("JuliaImageProcessing.jl");
include("JuliaOptimization.jl");
include("JuliaSignalProcessing.jl");

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions


## Parameters

# Data

numRowsA = 1000;
numColsA = 975;

numRowsK = 5;
numColsK = 4;

convMode = CONV_MODE_VALID;




## Load / Generate Data

mA = rand(numRowsA, numColsA);
mK = rand(numRowsK, numColsK);

if (convMode == CONV_MODE_FULL)
    mO = zeros((numRowsA, numColsA) .+ (numRowsK, numColsK) .- 1);
    hConv2D! = _Conv2D!;
elseif (convMode == CONV_MODE_VALID)
    mO = zeros((numRowsA, numColsA) .- (numRowsK, numColsK) .+ 1);
    hConv2D! = _Conv2DValid!;
end


## Analysis

@benchmark hConv2D!(mO, mA, mK)


## Display Results

