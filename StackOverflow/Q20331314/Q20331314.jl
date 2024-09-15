# StackOverflow Q20331314
# https://stackoverflow.com/questions/20331314
# Implementation of the Niblack Thresholding Algorithm.
# References:
#   1.  Michael Wirth - Niblack (https://craftofcoding.wordpress.com/2021/09/30/thresholding-algorithms-niblack-local).
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     15/09/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using ColorTypes;        #<! Required for Image Processing
using PlotlyJS;          #<! Use `add Kaleido_jll@v0.1` (https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using FileIO;            #<! Required for loading images
using LoopVectorization; #<! Required for Image Processing
using StableRNGs;
using StaticKernels;     #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images


## Settings

oRng    = StableRNG(RNG_SEED);

figureIdx       = 0;
exportFigures   = true;


## Functions

function ApplyNiblackThr( mI :: Matrix{T}, boxRadius :: N, ϵ :: T ) where {T <: AbstractFloat, N <: Integer}
    # Quantized image colors using K-Means.

    # Local Mean
    mM = BoxBlur(mI, boxRadius; padMode = PAD_MODE_REFLECT);
    # Local Variance
    # Based on: Var(x) = E[(x - μ)^2] = E[x^2] - E[x]^2
    mS = sqrt.(max.(BoxBlur(mI .* mI, boxRadius; padMode = PAD_MODE_REFLECT) .- (mM .* mM), zero(T)));

    return mI .> mM .+ ϵ .* mS; #<! Thresholding
    
end


## Parameters

# Data
# Resized version of the image in the post
imgUrl = raw"https://i.imgur.com/0ou5KzV.png"; #<! https://i.postimg.cc/BnCCfzjx/5aY03.png

# Niblack Algorithms
boxRadius   = 5;
ϵ           = -0.2;


#%% Load / Generate Data

mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = mI[:, :, 1];
mI = ScaleImg(mI);


# ## Analysis

mT = ApplyNiblackThr(mI, boxRadius, ϵ);


# ## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mT; titleStr = "Threshold Image, radius = $(boxRadius),  ϵ = $(ϵ)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
