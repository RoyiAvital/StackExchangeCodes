# StackOverflow Q2176266
# https://stackoverflow.com/questions/2176266
# Reduce (Quantize) Image Colors to 256 Colors.
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
using Clustering;
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

function QuantizeImageColors( mI :: Array{T, 3}, numColors :: N; numIter :: N = 1_000, ϵ :: T = 1e-6 ) where {T <: AbstractFloat, N <: Integer}
    # Quantized image colors using K-Means.

    numRows = size(mI, 1);
    numCols = size(mI, 2);
    
    numPx = numRows * numCols;

    # Generating the samples
    mX = permutedims(reshape(mI, (numPx, 3)), (2, 1)); #<! Shape: (3, numPx)

    # Clustering by the number of colors
    oKmeansRes = kmeans(mX, numColors; maxiter = numIter, tol = ϵ, display = :none);

    # Assigning each pixel to the cluster centroid color
    mC = oKmeansRes.centers[:, oKmeansRes.assignments];
    # Rebuild teh image from teh samples
    mO = reshape(permutedims(mC, (2, 1)), (numRows, numCols, 3));

    return mO;
    
end


## Parameters

# Data
imgUrl = raw"https://i.imgur.com/DAN1fMJ.png"; #<! Peppers Image (https://i.postimg.cc/TwZySw1w/peppers-trees.png)

# Quantization
numColors = 32;


#%% Load / Generate Data

mI = load(download(imgUrl)); #<! Mask
mI = ConvertJuliaImgArray(mI);
mI = ScaleImg(mI);


# ## Analysis

mQ = QuantizeImageColors(mI, numColors);


# ## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Source Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mQ; titleStr = "Quantized Image ($numColors Colors)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
