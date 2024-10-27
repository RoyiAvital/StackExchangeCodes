# StackExchange Signal Processing Q95393
# https://dsp.stackexchange.com/questions/95393
# Signal Reconstruction with Local Peaks Preservation.
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
# - 1.0.000     26/10/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;              #<! Required for Signal Processing
using ECOS;                #<! Required for Signal Processing
using LoopVectorization;   #<! Required for Image Processing
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image / Signal Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaSignalProcessing.jl")); #<! Signal Processing
include(joinpath(juliaCodePath, "JuliaSparseArrays.jl")); #<! Sparse Arrays
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

@enum DetrendMode begin
    DETREND_MODE_DC_BLOCKER
    DETREND_MODE_MEDIAN
end

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

## Parameters

dataUrl = "Data.csv"; #<! Local
# dataUrl = raw"https://github.com/RoyiAvital/StackExchangeCodes/raw/refs/heads/master/SignalProcessing/Q95393/Data.csv";

# Problem parameters

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

# Load the CSV data
vY = readdlm(dataUrl)[:]; #<! Data is a vector
# vY = readdlm(download(dataUrl))[:]; #<! Use when data is not local
numSamples = length(vY);


## Analysis

# The BEADS algorithm treat the filters matrices as commutative.  
# Which implies Circulant Matrices (Commutative discrete convolution).
# Circulant Matrices applies periodic convolution (Multiplication inf DFT).
# Hence assume both ends of the signal create some smoothness (Periodic).
# To have the assumption valid, the signal should be tapered.
# In this case, using Sigmoid function to have a roll off to zero.

# From `LoopVectorization.jl`
vYBegin = vY[1] * sigmoid_fast.(inv.(numSamplesTapering) * LinRange(-7.5 * numSamplesTapering, 7.5 * numSamplesTapering, numSamplesTapering));
vYEnd   = vY[end] * sigmoid_fast.(inv.(numSamplesTapering) * LinRange(7.5 * numSamplesTapering, -7.5 * numSamplesTapering, numSamplesTapering));
vYY = cat(vYBegin, vY, vYEnd; dims = 1);

numSamples = length(vYY);

vX, vF, vC = BeadsFilter(vYY, modelDeg, fₛ, asyRatio, λ₀, λ₁, λ₂, numIter; ϵ₀ = ϵ₀, ϵ₁ = ϵ₁);


## Display Results

# Display Data
figureIdx += 1;

titleStr = @sprintf("Beads Filter: d = %d, fc = %0.3f, r = %0.1f, λ₀ = %0.3f, λ₁ = %0.3f, λ₂ = %0.3f", modelDeg, fₛ, asyRatio, λ₀, λ₁, λ₂);

oTr1 = scatter(; x = 1:numSamples, y = vYY, mode = "markers", name = "Input Signal");
oTr2 = scatter(; x = 1:numSamples, y = vX, mode = "lines", name = "BEADS Signal");
oTr3 = scatter(; x = 1:numSamples, y = vF, mode = "lines", name = "BEADS Baseline");
oLayout = Layout(title = titleStr, width = 700, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.05, y = 0.975));
hP = Plot([oTr1, oTr2, oTr3], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


