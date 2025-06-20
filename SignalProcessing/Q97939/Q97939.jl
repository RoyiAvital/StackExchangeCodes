# StackExchange Signal Processing Q97939
# https://dsp.stackexchange.com/questions/97939
# Deriving the Batch Recursive Least Squares (Batch RLS) / Batch Sequential Least Squares with Limited Memory.
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
# - 1.0.000     20/06/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
# using BenchmarkTools;
# using Peaks;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function SqueezeArray( mA :: Array )
    
    tuSingletonDim = tuple((d for d in 1:ndims(mA) if size(mA, d) == 1)...);
    mAA = dropdims(mA; dims = tuSingletonDim);

    if ndims(mAA) == 0
        return mAA[1];
    else
        return mAA;
    end
end


function GenVandermondeMat!( mV :: Matrix{T}, vV :: Vector{T} ) where {T <: AbstractFloat}
    
    numRows = size(mV, 1);
    numCols = size(mV, 2);
    numRows == length(vV) || throw(DimensionMismatch("The number of rows of `mV` must match the number of elements in `vV`"));
    
    for ii = 1:numRows
        @inbounds mV[ii, 1] = one(T);
    end
    for jj = 2:numCols, ii = 1:numRows
        @inbounds mV[ii, jj] = vV[ii] * mV[ii, jj - 1];
    end
    
    return mV;

end

GenVandermondeMat( vV :: Vector{T}, numCols :: N) where {T <: AbstractFloat, N <: Integer} = GenVandermondeMat!(Matrix{T}(undef, length(vV), numCols), vV);

function SequentialLeastSquares!( vθ :: Vector{T}, vX :: Vector{T}, mR :: Matrix{T}, mH :: Matrix{T} ) where {T <: AbstractFloat}
    
    mR .-= mR * mH' * inv(I + mH * mR * mH') * mH * mR;
    mK   = mR * mH';
    vθ .+= mK * (vX - mH * vθ);

    return vθ, mR;

end

function SequentialLeastSquares( vθ :: Vector{T}, vX :: Vector{T}, mR :: Matrix{T}, mH :: Matrix{T} ) where {T <: AbstractFloat}

    mRR = mR - mR * mH' * inv(I + mH * mR * mH') * mH * mR;
    mK  = mRR * mH';
    vθθ = vθ + mK * (vX - mH * vθ);

    return vθθ, mRR;

end

## Parameters

modelOrder     = 3;
numSamples     = 35;
numSamplesInit = 5;
batchSize      = 5;
σ              = 2;

## Load / Generate Data

# Load the Signal

vH = collect(LinRange(0.0, 1.0, numSamples)); #<! Grid
mH = GenVandermondeMat(vH, modelOrder + 1); #<! Model Matrix
vθ = 3 * randn(oRng, modelOrder + 1); #<! Parameters (Ground truth)
vN = σ * randn(oRng, numSamples);
vY = mH * vθ; #<! Model Data
vX = vY + vN; #<! Measurements

## Analysis

vθLs  = mH \ vX; #<! LS Estimation
vθSls = mH[1:numSamplesInit, :] \ vX[1:numSamplesInit]; #<! Sequential LS initialization
mR    = inv(mH[1:numSamplesInit, :]' * mH[1:numSamplesInit, :]); #<! Sequential LS initialization

## Display Results

# Display Data
figureIdx += 1;

sTr1    = scatter(; x = vH, y = vY, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "Model", name = "Model");
sTr2    = scatter(; x = vH, y = vX, mode = "markers", 
                  line = attr(width = 2.0),
                  text = "Measurements", name = "Measurements");
sTr3    = scatter(; x = vH, y = mH * vθLs, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "LS Estimation", name = "LS Estimation");
sLayout = Layout(title = "The Model, Measurements ($(numSamples)) and Estimation", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Grid",
                 yaxis_range = [-5.0, 5.0],
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

# Display Sequential LS initialization

numBatch = 1;

# Display Data
figureIdx += 1;

sTr1    = scatter(; x = vH, y = vX, mode = "markers", 
                  line = attr(width = 2.0),
                  text = "Measurements", name = "Measurements");
sTr2    = scatter(; x = vH, y = mH * vθLs, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "LS Estimation", name = "LS Estimation");
sTr3    = scatter(; x = vH, y = mH * vθSls, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "Sequential LS Estimation", name = "Sequential LS Estimation");
sLayout = Layout(title = "The Sequential LS with Batch Size of $(batchSize) on Iteration: $(numBatch)", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Grid",
                 yaxis_range = [-5.0, 5.0],
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

# Display Sequential LS

for ii = (numSamplesInit + 1):batchSize:numSamples
    global mR;
    global vθSls;
    global numBatch += 1;
    mHH = mH[ii:(ii + batchSize - 1), :];
    vXX = vX[ii:(ii + batchSize - 1)];

    vθSls, mR = SequentialLeastSquares(vθSls, vXX, mR, mHH);

global figureIdx += 1;

sTr1    = scatter(; x = vH, y = vX, mode = "markers", 
                  line = attr(width = 2.0),
                  text = "Measurements", name = "Measurements");
sTr2    = scatter(; x = vH, y = mH * vθLs, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "LS Estimation", name = "LS Estimation");
sTr3    = scatter(; x = vH, y = mH * vθSls, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "Sequential LS Estimation", name = "Sequential LS Estimation");
sLayout = Layout(title = "The Sequential LS with Batch Size of $(batchSize) on Iteration: $(numBatch)", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Grid",
                 yaxis_range = [-5.0, 5.0],
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

end

# Generate the Animation
# 1. Download APNG Assembler.
# 2. Delete the first figure (`Figure0001.png`).
# 3. Run on command line: `apngasm64 out.png Figure0002.png 7 8 -l0`.