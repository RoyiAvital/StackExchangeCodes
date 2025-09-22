# StackExchange Mathematics Q2609510
# https://math.stackexchange.com/questions/2609510
# Calculating the Gap of SVM Classifier Given a Set of Points and the Separating Hyperplane Direction.
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
# - 1.0.000     23/09/2025  Royi Avital
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
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( mX :: Matrix{T}, vY :: Vector{N}, vW :: Vector{T}, valB :: T ) where {T <: AbstractFloat, N <: Integer}
    # Per class, calculate the minimum distance.
    # The non negative objective to minimize is the absolute difference.
    # It is zero where `valB` is set such that the line is in the middle of the support vectors.
    # Assume Binary SVM.

    # Minimum distance per class (Binary -> 2 Classes)
    vD = 1e9 * ones(T, 2); #<! Minimum distance per class (Assumes less than 1e9)
    numSamples = size(mX, 2);

    for ii in 1:numSamples
        vX = mX[:, ii];
        clsIdx = ifelse(vY[ii] == -1, 1, 2); #<! (-1) -> 1, (1) - > 2
        valD = DistToPlane(vX, vW, valB);
        if valD < vD[clsIdx]
            vD[clsIdx] = valD;
        end
    end
    
    return vD[1] - vD[2];
    
end

function DistToPlane( vP :: Vector{T}, vW :: Vector{T}, valB :: T ) where {T <: AbstractFloat}

    return abs(dot(vW, vP) + valB) / norm(vW);
    
end


## Parameters

# Data
vW = [4.0, 3.0, 0.0, 0.0];
tuGrid = (-5.0, 5.0, 2_001); #<! (left boundary, right boundary, number of grid points)

# Solver
numIter = 10_000;
η       = 1e-5; #<! Step Size


## Load / Generate Data

mX = [[1.0, 1.0, 0.0, 1.0];; [1.0, 1.0, 0.0, −1.0];; [−1.0, 1.0, 0.0, 1.0];; [1.0, −2.0, 0.0, 1.0]]; #<! Each sample is a column
vY = [1, 1, -1, -1];

numSamples = size(mX, 2);

## Analysis

hObjFun(valB :: T) where {T <: AbstractFloat} = ObjFun(mX, vY, vW, valB);

vG = collect(LinRange(tuGrid...));
vB = abs.(hObjFun.(vG));
valB = FindZeroBinarySearch(hObjFun, tuGrid[1], tuGrid[2]);


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, 2);

vTr[1] = scatter(x = vG, y = vB, mode = "lines",
                 line = attr(width = 3.0),
                 text = "Objective Value", name = "Objective Value")
vTr[2] = scatter(x = [valB], y = [hObjFun(valB)], mode = "markers",
                 marker = attr(size = 10),
                 text = "Optimal Value", name = "Optimal Value")

vShp = [vline(valB, line_color = "red", line_dash = "dashdot", line_width = 3.0)];
optValStr = @sprintf("%0.2f", valB);
vAnn = [attr(x = valB + 0.15, y = 1.0, text = optValStr, 
             xanchor = "left", yanchor = "bottom", showarrow = false,
             font_color = "red")];

sLayout = Layout(title = "Objective Function", width = 600, height = 600, 
                 xaxis_title = "b", yaxis_title = "Objective Value",
                 xaxis_range = (tuGrid[1], tuGrid[2]),
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99),
                 shapes = vShp, annotations = vAnn);

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

