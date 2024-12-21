# StackExchange Signal Processing Q95814
# https://dsp.stackexchange.com/questions/95814
# Interpolate Image Pixels based on Pixel Coordinate Distance from Color Blobs.
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
# - 1.0.001     21/12/2024  Royi Avital
#   *   Added support for RGBA images.
# - 1.0.000     16/12/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using ColorTypes;          #<! Required for Image Processing
# using Downloads;
using FileIO;              #<! Required for loading images
using LoopVectorization;   #<! Required for Image Processing
# using MAT;
using NearestNeighbors;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image / Signal Processing
using StatsBase;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaArrays.jl")); #<! Sparse Arrays
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl")); #<! Display Images
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function UniqueColorsMap( mI :: Array{T, 3} ) where {T <: AbstractFloat}

    mT = permutedims(mI, (3, 1, 2));
    mT = reinterpret(reshape, NTuple{3, T}, mT);
    dColorMap = Dict{eltype(mT), Vector{Int}}();

    for ii in 1:length(mT)
        vI = get!(dColorMap, mT[ii]) do
            Vector{Int}[];
        end
        push!(vI, ii);
    end

    return dColorMap;

end

function UniqueColorsMap( mI :: Matrix{T} ) where {T <: AbstractFloat}

    dColorMap = Dict{eltype(mI), Vector{Int}}();

    for ii in 1:length(mI)
        vI = get!(dColorMap, mI[ii]) do
            Vector{Int}[];
        end
        push!(vI, ii);
    end

    return dColorMap;

end

function LinearSubScripts( linIdx :: N, numRows :: N, numCols :: N ) where {N <: Integer}
    # Calculate row and column indices
    ii = (linIdx - 1) % numRows + 1;   #<! Row index
    jj = div(linIdx - 1, numRows) + 1;  #<! Column index
    return ii, jj;
end

function GetColorInd( vI :: Vector{N}, numRows :: N, numCols :: N ) where {N <: Integer}

    # Coordinates
    numPx   = length(vI);
    mC      = zeros(Int, 2, numPx);
    pxIdx = 1
    for ii = 1:numPx
        ii, jj = LinearSubScripts(vI[ii], numRows, numCols);
        mC[1, pxIdx] = ii;
        mC[2, pxIdx] = jj;
        pxIdx += 1;
    end

    return mC;
    
end

## Parameters

imgFileName = "fh5cmfav.png"
imgUrl      = raw"https://i.sstatic.net/z1FZ6ea5.png"; #<! https://i.imgur.com/lonCgZl.png

# Problem parameters
σ = 7.5;

## Load / Generate Data

# Load the Image
if !isfile(imgFileName)
    download(imgUrl, imgFileName);
end
mI = load(imgFileName); #<! Local
mI = ConvertJuliaImgArray(mI);
if (size(mI, 3) > 3)
    # Drop Alpha channel
    mI = mI[:, :, 1:3];
end
mI = ScaleImg(mI);

mO = copy(mI); #<! Output

## Analysis

numRows = size(mI, 1);
numCols = size(mI, 2);

# Mask
mB = dropdims(any(mI .> 0; dims = 3); dims = 3);

dColorMap = UniqueColorsMap(mI);
delete!(dColorMap, (0.0, 0.0, 0.0));

numUniqueColors = length(dColorMap);

# Tree for Coordinate Distances
dColorKDTree = Dict{NTuple{3, Float64}, Any}();
for (key, val) in pairs(dColorMap)
    mC = GetColorInd(val, numRows, numCols);
    push!(dColorKDTree, key => KDTree(Float64.(mC)));
end

lKeys = collect(keys(dColorMap)); #<! List of keys

# Calculate the distance per pixel from closest indices of each color
vP = zeros(2);
mW = zeros(numRows, numCols, numUniqueColors);
for jj in 1:numCols
    for ii in 1:numRows
        vP[1] = Float64(ii);
        vP[2] = Float64(jj);
        for kk in 1:numUniqueColors
            colorIdx, colDistance = nn(dColorKDTree[lKeys[kk]], vP); #<! Nearest Neighbor
            mW[ii, jj, kk] = colDistance;
        end
    end
end

# Normalize the weights
mW = exp.(-0.5 .* ((mW ./ σ) .^ 2));
mW ./= dropdims(sum(mW; dims = 3); dims = 3);

# Interpolate the output image
for jj in 1:numCols
    for ii in 1:numRows
        for kk in 1:numUniqueColors
            if mB[ii, jj] #<! Value exists
                continue;
            end
            mO[ii, jj, :] .+= mW[ii, jj, kk] .* lKeys[kk];
        end
    end
end

## Display Results

# Display Data
figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image")
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image, σ = $(σ)")
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
