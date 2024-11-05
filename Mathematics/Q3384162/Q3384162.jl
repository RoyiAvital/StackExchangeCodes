# StackExchange Mathematics Q3384162
# https://math.stackexchange.com/questions/3384162
# Remove Drift from a Signal.
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
# - 1.0.000     05/11/2024  Royi Avital
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
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StatsBase;
using StaticKernels;       #<! Required for Signal Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaSignalProcessing.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions


## Parameters

# Data
tuGridSupport   = (0, 10);
numGridPts      = 1000;
numSections     = 5;

sinAmp   = 0.35;
sineFreq = 0.75;
noiseStd = 0.25;

# Model
polyDeg = 1;
λ       = 505.5;
vλ      = LinRange(0.1, 10.0, 5);
ρ       = 199.5; #<! Should be proportional to λ

# Solvers
numIterations = Unsigned(50_000);


## Load / Generate Data

numSamples = numGridPts;

vG = LinRange(tuGridSupport[1], tuGridSupport[2], numGridPts);
# mG = [vG[ii] ^ jj for ii in 1:numGridPts, jj in 0:1]; #<! Matrix model for affine function
mG = vG .^ (0:1)'; #<! Matrix model for affine function
# mP = randn(oRng, 2, numSections); #<! Parameters of linear function
# mP = mP[:, sortperm(mP[1, :])]; #<! First column a, second b : y_i = a x_i + b
# vY = vec(maximum(mG * mP; dims = 2) + (noiseStd * randn(oRng, numGridPts)));

vP = sort(5 * randn(oRng, numSections)); #!< Values
vP = [-7, 5, -4, 9, -2]; #!< Values
vI = reduce(vcat, [1, sample(oRng, 10:(numGridPts - 10), numSections - 1; replace = false, ordered = true), numGridPts]); #<! Break index
vI = [1, 188, 407, 570, 810, numGridPts]; #<! Break index
vY = zeros(numGridPts);

for ii in 1:numSections
    vY[vI[ii]:vI[ii + 1]] .= vP[ii];
end
vY = cumsum(vY);
vY = 5. * (vY ./ maximum(abs.(vY)));
vY += sinAmp * sin.(2π * sineFreq * vG); #<! Sine component
vY .+= noiseStd * randn(oRng, numGridPts);


## Analysis

vX = L1PieceWise(vY, λ, polyDeg);


## Display Analysis

figureIdx += 1;

oTr = scatter(x = collect(vG), y = vY, 
                mode = "markers", text = "Data Samples", name = "Data Samples");
oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr1 = scatter(x = vG, y = vY, 
                mode = "markers", text = "Data Samples", name = "Data Samples");
oTr2 = scatter(x = vG, y = vX, 
                mode = "lines", text = "Model", name = "Model");
oLayout = Layout(title = "Data Samples and Piece Wise Linear Model", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y",
                 legend = attr(x = 0.01, y = 0.99));

hP = plot([oTr1, oTr2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

