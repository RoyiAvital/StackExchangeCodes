# StackOverflow Q11473503
# https://stackoverflow.com/questions/11473503
# Efficient Implementation of the Kuwahara Filter.
# References:
#   1.  Wikipedia - Kuwahara Filter (https://en.wikipedia.org/wiki/Kuwahara_filter).
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     16/09/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
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

function KuwaharaFilter( mI :: Matrix{T}, filtRadius :: N ) where {T <: AbstractFloat, N <: Integer}
    # The radius is the radius of the sub quadrant.
    # If the radius is `r` then the filter size is `2 * (2r) + 1`.
    
    numRows, numCols = size(mI);
    
    # kuwaharaFilterRadius = N(2) * filtRadius;
    mP = PadArray(mI, filtRadius, PAD_MODE_REFLECT);

    # Local Mean
    mM = BoxBlur(mP, filtRadius; padMode = PAD_MODE_REFLECT);
    # Local Variance (Clipped to a non negative value)
    # Based on: Var(x) = E[(x - μ)^2] = E[x^2] - E[x]^2
    mV = max.(BoxBlur(mP .* mP, filtRadius; padMode = PAD_MODE_REFLECT) .- (mM .* mM), zero(T));

    mO = similar(mI);

    for jj ∈ 1:numCols, ii ∈ 1:numRows
        rr = ii + filtRadius;
        cc = jj + filtRadius;

        stdArg = argmin((mV[rr - filtRadius, cc - filtRadius], mV[rr - filtRadius, cc + filtRadius], mV[rr + filtRadius, cc + filtRadius], mV[rr + filtRadius, cc - filtRadius]));
        mO[ii, jj] = (mM[rr - filtRadius, cc - filtRadius], mM[rr - filtRadius, cc + filtRadius], mM[rr + filtRadius, cc + filtRadius], mM[rr + filtRadius, cc - filtRadius])[stdArg];
    end

    return mO;
    
end

function KuwaharaFilter( mI :: Array{T, 3}, filtRadius :: N ) where {T <: AbstractFloat, N <: Integer}

    mO = similar(mI);

    for ii ∈ 1:3
        mO[:, :, ii] = KuwaharaFilter(mI[:, :, ii], filtRadius);
    end

    return mO;

end


## Parameters

# Data
imgUrl = raw"https://i.imgur.com/DAN1fMJ.png"; #<! Peppers Image (https://i.postimg.cc/TwZySw1w/peppers-trees.png)

# Niblack Algorithms
filtRadius = 4;


#%% Load / Generate Data

mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = ScaleImg(mI);

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx   = numRows *  numCols;


# ## Analysis

mO = KuwaharaFilter(mI, filtRadius);

runTime = @belapsed KuwaharaFilter($mI, 3);
@printf("The Filter Runtime: %0.2f [Mili Sec]", 1000.0 * runTime);
@printf("The Filter Runtime for a Mega Pixel: %0.2f [Sec / Mega Pixel]", 1e6 * (runTime / numPx));


# ## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image, radius = $(filtRadius)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
