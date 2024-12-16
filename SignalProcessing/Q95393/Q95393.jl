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
using DSP;
using ECOS;                #<! Required for Signal Processing
using LoopVectorization;   #<! Required for Image Processing
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
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

paramK      = 3; #<! Derivative order
localRadius = 4; #<! Radius
detrendMode = DETREND_MODE_DC_BLOCKER;
medRadius   = 25; #<! Radius for Median Filter
α           = 0.995; #<! DC Blocker
λ           = 500; #<! Smoothing balance


## Load / Generate Data

# Load the CSV data
vY = readdlm(dataUrl)[:]; #<! Data is a vector
# vY = readdlm(download(dataUrl))[:]; #<! Use when data is not local
numSamples = length(vY);


## Analysis

# Local Peaks
numElementsN = 2 * localRadius + 1; #<! Number of elements in the window
vO = OrderFilter(vY, localRadius, numElementsN - (floor(Int, sqrt(localRadius)) + 1) + 1);
vP = vY .>= vO;

# Build the Finite Differences Operator
mD  = spdiagm(numSamples, numSamples, 0 => -ones(numSamples), 1 => ones(numSamples - 1));
mDD = copy(mD);
for kk in 1:(paramK - 1)
    mD[:] = mD * mDD;
end
mD = mD[1:(end - paramK), :];

# Trend and Detrend Signal
if (detrendMode == DETREND_MODE_DC_BLOCKER)
    vB = [1.0, -1.0];
    vA = [1.0, -α];
    
    # The `filtfilt(vB, vA, vY)` does not work (https://github.com/JuliaDSP/DSP.jl/issues/573)
    vZ = filtfilt(PolynomialRatio(vB, vA), vY);
    vT = vY - vZ;
elseif (detrendMode == DETREND_MODE_MEDIAN)
    vT = MedianFilter(vY, medRadius);
    vZ = vY - vT;
end

# Solving the Convex Problem
vPi = findall(vP); #<! Convex.jl does not support boolean indexing (https://github.com/jump-dev/Convex.jl/issues/707)

vX = Variable(numSamples);
sConvProb = minimize( 0.5 * sumsquares(vX - vZ) + λ * norm(mD * vX, 1), [vX[vPi] == vY[vPi]] );
solve!(sConvProb, ECOS.Optimizer; silent = true);
vX = vec(vX.value);


## Display Results

# Display Data
figureIdx += 1;

oTr = scatter(; x = 1:numSamples, y = vY, mode = "lines+markers");
oLayout = Layout(title = "Input Signal", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Display the Local Peak Set
figureIdx += 1;

oTr1 = scatter(; x = 1:numSamples, y = vY, mode = "lines+markers", name = "Signal");
oTr2 = scatter(; x = (1:numSamples)[vP], y = vY[vP], mode = "markers", name = "Local Peak Set");
oLayout = Layout(title = "Input Signal", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Display the Trend Signal
figureIdx += 1;

if (detrendMode == DETREND_MODE_DC_BLOCKER)
    detrendStr = @sprintf(", Detrend by DC Blocker with α = %0.2f", α);
elseif (detrendMode == DETREND_MODE_MEDIAN)
    detrendStr = @sprintf(", Detrend by Median Filter with radius = %d", medRadius);
end

titleStr = "Input & Trend Signals" * detrendStr;

oTr1 = scatter(; x = 1:numSamples, y = vY, mode = "lines", name = "Signal");
oTr2 = scatter(; x = 1:numSamples, y = vT, mode = "lines", name = "Trend");
oTr3 = scatter(; x = 1:numSamples, y = vZ, mode = "lines", name = "DeTrend");
oLayout = Layout(title = titleStr, width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2, oTr3], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Display Reconstruction
figureIdx += 1;

titleStr = "Reconstructed Signal" * detrendStr;

oTr1 = scatter(; x = 1:numSamples, y = vY, mode = "lines", name = "Signal");
oTr2 = scatter(; x = (1:numSamples)[vP], y = vY[vP], mode = "markers", name = "Local Peak Set");
oTr3 = scatter(; x = 1:numSamples, y = vX, mode = "lines", name = "Reconstruction");
oLayout = Layout(title = titleStr, width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2, oTr3], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


