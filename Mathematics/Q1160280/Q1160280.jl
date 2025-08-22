# StackExchange Mathematics Q1160280
# https://math.stackexchange.com/questions/1160280
# Maximum Likelihood Estimator of Polynomial Ratio Distribution.
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
# - 1.0.000     22/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function CalcDistribution( vY :: Vector{T}, α :: T ) where {T <: AbstractFloat}

    vD = T(5) .* (α ^ 5) ./ (vY .^ 6);

    return vD;
    
end


## Parameters

# Data
numGridPts = 1_000;
maxY       = 1.0;
vα         = collect(LinRange(0.01, 0.9, 6));


## Load / Generate Data

vY = LinRange(0, maxY, numGridPts);
vY = collect(vY[2:numGridPts]);

numα = length(vα);
mD   = zeros(length(vY), numα);


## Analysis

for (ii, α) in enumerate(vα)
    mD[:, ii] = CalcDistribution(vY, α);
end


## Display Results

maxVal = maximum(mD);
mD   .*= (0.97 / maxVal);

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, numα);

for (ii, α) in enumerate(vα)
    shiftVal = ii - 1.0;
    nameStr = @sprintf("%0.2f", α);
    vTr[ii] = scatter(; x = vY, y = shiftVal .+ mD[:, ii], mode = "lines", 
               line_width = 1.5,
               name = nameStr, text = nameStr);
end

sLayout = Layout(title = "Likelihood Function", width = 600, height = 600, 
                 xaxis_title = "y", yaxis_title = "PDF",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

