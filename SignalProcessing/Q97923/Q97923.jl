# StackExchange Signal Processing Q97923
# https://dsp.stackexchange.com/questions/97923
# Robust Extraction of Local Peaks in Noisy Signal with a Trend.
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
# - 1.0.000     21/06/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
# using BenchmarkTools;
using Convex;              #<! Required for Signal Processing
using ECOS;                #<! Required for Signal Processing
using LoopVectorization;   #<! Required for Image Processing
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;        #<! Required for Arrays
using StableRNGs;
using StaticKernels;       #<! Required for Image / Signal Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaArrays.jl")); #<! Sparse Arrays
include(joinpath(juliaCodePath, "JuliaSignalProcessing.jl")); #<! Signal Processing
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions


## Parameters

# Data
dataUrl = "Signal.csv"; #<! Local
# dataUrl = raw"https://github.com/RoyiAvital/StackExchangeCodes/raw/refs/heads/master/SignalProcessing/Q97923/Signal.csv";

# BEDS Algorithm
modelDeg    = 1; #<! Derivative order
fₛ           = 0.015; #<! Cut Off Frequency
asyRatio    = 5.0 #<! Positive / Negative Peak Ratio
λ₀          = 0.25;
λ₁          = 0.035;
λ₂          = 0.015;
numIter     = 31;
ϵ₀          = 5e-5;
ϵ₁          = 5e-5;

numSamplesTapering = 50;

## Load / Generate Data

# Load the Signal
mD = readdlm(dataUrl, ';'; skipstart = 1); #<! Data is a vector
vX = mD[:, 1]; #<! Sampling grid
vY = mD[:, 2]; #<! Samples


## Analysis

# The BEADS algorithm treat the filters matrices as commutative.  
# Which implies Circulant Matrices (Commutative discrete convolution).
# Circulant Matrices applies periodic convolution (Multiplication of DFT's).
# Hence assume both ends of the signal create some smoothness (Periodic).
# To have the assumption valid, the signal should be tapered.
# In this case, using Sigmoid function to have a roll off to zero.

vXX, vFF, vCC = BeadsFilter(vY, modelDeg, fₛ, asyRatio, λ₀, λ₁, λ₂, numIter; ϵ₀ = ϵ₀, ϵ₁ = ϵ₁);

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