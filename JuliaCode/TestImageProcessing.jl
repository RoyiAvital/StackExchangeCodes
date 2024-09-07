# Test Code for Image Processing
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
# - 1.0.000     07/09/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using BenchmarkTools;
using MAT;
using UnicodePlots;

## Constants & Configuration

## External
include("JuliaInit.jl");
include("JuliaImageProcessing.jl");

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions


## Parameters

# Data
matFileName = "TestImageProcessing.mat";

dPadMode = Dict("circular" => PAD_MODE_CIRCULAR, "constant" => PAD_MODE_CONSTANT, "replicate" => PAD_MODE_REPLICATE, "symmetric" => PAD_MODE_SYMMETRIC);
dConvMode = Dict("full" => CONV_MODE_FULL, "same" => CONV_MODE_SAME, "valid" => CONV_MODE_VALID);


## Load / Generate Data

dMatData = matread(matFileName);
mI = dMatData["mI"];
numRows, numCols = size(mI);


## Analysis

# Pad Array
cPad = dMatData["cPad"]; #<! Cell array is mapped to a vector of `Any` in Julia
numTests = size(cPad, 1);

println("Pad Array Tests")
for ii ∈ 1:numTests
    tuPadSize   = tuple(Int.(cPad[ii, 2])...);
    padMode     = dPadMode[cPad[ii, 3]];
    
    mO = PadArray(mI, tuPadSize, padMode);
    
    maxErr = maximum(abs.(mO - cPad[ii, 1]));
    @printf("Iteration: %04d, Pad Size: (%d, %d), Pad Mode: %s, Maximum Error: %0.3f\n", ii, tuPadSize[1], tuPadSize[2], cPad[ii, 3], maxErr);
end

println("")

# Convolution 2D
cConv = dMatData["cConv"]; #<! Cell array is mapped to a vector of `Any` in Julia
numTests = size(cConv, 1);

println("Convolution 2D Tests")
for ii ∈ 1:numTests
    tuKerSize   = tuple(Int.(cConv[ii, 2])...);
    # Handle the case the MATLAB matrix is 1x1 which is interpreted in Julia as a scalar
    mK          = Matrix{Float64}(undef, tuKerSize);
    mK[:]       = collect(cConv[ii, 3])[:];
    convMode    = dConvMode[cConv[ii, 4]];

    mO = Conv2D(mI, mK; convMode = convMode);
    
    maxErr = maximum(abs.(mO - cConv[ii, 1]));
    @printf("Iteration: %04d, Kernel Size: (%d, %d), Conv Mode: %s, Maximum Error: %0.25f\n", ii, tuKerSize[1], tuKerSize[2], cConv[ii, 4], maxErr);
end



## Display Results

