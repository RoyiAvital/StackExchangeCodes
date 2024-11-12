# StackExchange Mathematics Q722698
# https://math.stackexchange.com/questions/722698
# 2D Localization by Distance Measurements.
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
# - 1.0.000     03/11/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

@enum NoiseModel begin
    GAUSSIAN
    EXPONENTIAL
end


## Functions


## Parameters

# Data
numMeasurements = 7;
noiseModel      = GAUSSIAN;
noiseFactor     = 0.1;
numOutliers     = 2; #<! Must be <= `numMeasurements`
tuGridSize      = (2, 2); #<! x, y

# Solver / Solution
numGridPts = 100; #<! Per dimension

# Solver
numIter = 1_000;
η       = 1e-3;

# Visualization


## Load / Generate Data

vPtLocation = [rand(oRng) * tuGridSize[1], rand(oRng) * tuGridSize[2]];
# 2D Array Comprehension in Julia: https://discourse.julialang.org/t/75659
mAccPtLoc   = [rand(oRng) * tuGridSize[jj] for ii ∈ 1:numMeasurements, jj ∈ 1:length(tuGridSize)];
# TODO: Add support for other noise models
# TODO: Add support for outliers
vMeasNoise  = noiseFactor * randn(oRng, numMeasurements);
vDist       = [norm(mAccPtLoc[ii, :] - vPtLocation) for ii ∈ 1:numMeasurements];
vMeasure    = vMeasNoise + vDist;


## Analysis

## Display Analysis

figureIdx += 1;

oTr1 = scatter(; x = mAccPtLoc[:, 1], y = mAccPtLoc[:, 2], mode = "markers", 
                marker_size = 12, text = "Access Point", name = "Access Points");
oTr2 = scatter(; x = [vPtLocation[1]], y = [vPtLocation[2]], mode = "markers", 
                marker_size = 12, text = "Reference Point", name = "Reference Point");
oShp = [circle(x0 = mAccPtLoc[ii, 1] - vMeasure[ii], y0 = mAccPtLoc[ii, 2] - vMeasure[ii], 
                x1 = mAccPtLoc[ii, 1] + vMeasure[ii], y1 = mAccPtLoc[ii, 2] + vMeasure[ii];
                opacity = 0.15, fillcolor = "black", line_color = "white") for ii ∈ 1:numMeasurements]
oLayout = Layout(title = "Localization by Range Measurements: Scenario", width = 600, height = 600, 
                xaxis_range = [0, tuGridSize[1]], yaxis_range = [0, tuGridSize[2]], xaxis_title = 'x', yaxis_title = 'y',
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                legend = attr(x = 0.025, y = 0.975), shapes = oShp);
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


