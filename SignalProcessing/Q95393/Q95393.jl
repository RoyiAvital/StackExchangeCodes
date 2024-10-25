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
using LoopVectorization;   #<! Required for Image Processing
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images
include(joinpath(juliaCodePath, "JuliaSparseArrays.jl")); #<! Sparse Arrays


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

## Parameters

dataUrl = raw"https://i.postimg.cc/85Jjs9wJ/Flowers.png"; #<! https://i.imgur.com/PckT6jF.png
dataUrl = "Data.csv";

# Problem parameters

paramK = 2; #<! Radius (The K in the paper K = 2k + 1)


## Load / Generate Data

vA = readdlm(dataUrl)[:]; #<! Data is a vector
numSamples = length(vA);


# mI = load(download(imgUrl));

# Display Data

oTr = scatter(; x = 1:numSamples, y = vA, mode = "lines+markers");
oLayout = Layout(title = "Input Signal", width = 600, height = 400, hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);


## Analysis



## Display Results


