# Test Code for Image Processing
# Several tests for the Julia Code based on Python Reference.
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
matFileName = "TestImageProcessingPython.mat";

dBndMode    = Dict("fill" => PAD_MODE_CONSTANT, "wrap" => PAD_MODE_CIRCULAR, "symm" => PAD_MODE_SYMMETRIC); #<! `scipy.signal.convolve2d()`
dConvMode   = Dict("full" => CONV_MODE_FULL, "same" => CONV_MODE_SAME, "valid" => CONV_MODE_VALID); #<! `scipy.signal.convolve2d()`
dGaussMode  = Dict("constant" => PAD_MODE_CONSTANT, "mirror" => PAD_MODE_REFLECT, "nearest" => PAD_MODE_REPLICATE, "reflect" => PAD_MODE_SYMMETRIC, "wrap" => PAD_MODE_CIRCULAR); #<! `scipy.ndimage.gaussian_filter()`


## Load / Generate Data

dMatData = matread(matFileName);
mI = dMatData["mI"];
numRows, numCols = size(mI);


## Analysis

# Convolution 2D - `scipy.signal.convolve2d()`
cConv = dMatData["cConv"]; #<! Cell array is mapped to a vector of `Any` in Julia
numTests = size(cConv, 1);

println("Convolution 2D Tests")
for ii ∈ 1:numTests
    tuKerSize   = tuple(cConv[ii, 2]...);
    tuKerRad    = tuple(tuKerSize[1] ÷ 2, tuKerSize[2] ÷ 2);
    # Handle the case the Numpy matrix is 1x1 which is interpreted in Julia as a scalar
    mK          = Matrix{Float64}(undef, tuKerSize);
    mK[:]       = collect(cConv[ii, 3])[:];
    padMode     = dBndMode[cConv[ii, 5]];
    convMode    = dConvMode[cConv[ii, 4]];

    # Python use the `convMode` to set the output size.
    # For `CONV_MODE_VALID` the padding is meaningless.
    # For `CONV_MODE_FULL` the padding should be doubled so the output has `full` size.
    # For `CONV_MODE_SAME` the padding should as defined and the output `valid`.
    # TODO: Handle the case of a kernel with even size.
    if (convMode == CONV_MODE_VALID)
        mIPad = copy(mI);
    elseif (convMode == CONV_MODE_FULL)
        mIPad = PadArray(mI, 2 .* tuKerRad, padMode);
    elseif (convMode == CONV_MODE_SAME)
        mIPad = PadArray(mI, tuKerRad, padMode);
    end

    mO = Conv2D(mIPad, mK; convMode = CONV_MODE_VALID);
    
    maxErr = maximum(abs.(mO - cConv[ii, 1]));
    @printf("Iteration: %04d, Kernel Size: (%d, %d), Pad Mode: %s, Conv Mode: %s, Maximum Error: %0.25f\n", ii, tuKerSize[1], tuKerSize[2], cConv[ii, 5], cConv[ii, 4], maxErr);
end

println("")

# Gaussian Filter - `scipy.ndimage.gaussian_filter()`
cGauss = dMatData["cGauss"]; #<! Cell array is mapped to a vector of `Any` in Julia
numTests = size(cGauss, 1);

println("Gaussian Convolution 2D Tests")
for ii ∈ 1:numTests
    kernelStd   = cGauss[ii, 2];
    kernelRad   = cGauss[ii, 3];
    mK          = GenGaussianKernel(kernelStd, (kernelRad, kernelRad));
    # The mapping of `gaussian_filter()` is strange.
    # "mirror" is actually "reflect" and "reflect" is actually "symmetric".
    padMode     = dGaussMode[cGauss[ii, 4]];

    mIPad = PadArray(mI, (kernelRad, kernelRad), padMode);
    mO = Conv2D(mIPad, mK; convMode = CONV_MODE_VALID);
    
    maxErr = maximum(abs.(mO - cGauss[ii, 1]));
    @printf("Iteration: %04d, Kernel STD: %0.3f, Kernel Radius: %d, Pad Mode: %s, Maximum Error: %0.25f\n", ii, kernelStd, kernelRad, cGauss[ii, 4], maxErr);
end

println("")



## Display Results

