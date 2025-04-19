# StackExchange Computational Science Q45000
# https://scicomp.stackexchange.com/questions/45000
# The Minimum Area Enclosing Circle.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     19/04/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
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


## Parameters

# Data
numSamples = 100;


#%% Load / Generate Data

mX = randn(oRng, 2, numSamples);


## Analysis

dataDim = size(mX, 1);

vC = Variable(dataDim);
circRadius = Variable(1);
sConvProb = minimize( circRadius, [Convex.norm(vX - vC) <= circRadius for vX in eachcol(mX)] );
solve!(sConvProb, ECOS.Optimizer; silent = true);


vC         = vec(vC.value);
circRadius = circRadius.value;


## Display Results

figureIdx += 1;

oTr1 = scatter(x = mX[1, :], y = mX[2, :], mode = "markers", name = "Date Samples");
oTr2 = scatter(x = [vC[1]], y = [vC[2]], mode = "markers", name = "Circle Center");

vShp = [circle(x0 = vC[1] - circRadius, y0 = vC[2] - circRadius, 
                x1 = vC[1] + circRadius, y1 = vC[2] + circRadius;
                opacity = 0.15, fillcolor = "red", line_color = "red")]

oLayout = Layout(title = "Date Samples and Enclosing Circle", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "x", yaxis_title = "y", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975), shapes = vShp);

vTr = [oTr1, oTr2];
hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end