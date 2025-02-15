# StackExchange Mathematics Q1193773
# https://math.stackexchange.com/questions/1193773
# Practical Exercise in SVM for 1D Data.
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
# - 1.0.000     15/02/2025  Royi Avital
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
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
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

function SolveSoftSVM1D( vX :: Vector{T}, vY :: Vector{N}, paramC :: T ) where {T <: AbstractFloat, N <: Integer}
    # SVM Formulations:
    # - https://stats.stackexchange.com/questions/213687
    # - https://stats.stackexchange.com/questions/198256

    numPts = length(vX);

    if length(vY) != numPts
        throw(DimensionMismatch(lazy"`vY`` has length $(length(vY)), `vX` has length $(length(vX))"));
    end

    paramW = Variable();
    paramB = Variable();
    vξ     = Variable(numPts);

    vConst = [T(vY[ii]) * (paramW * vX[ii] + paramB) >= 1.0 - vξ[ii] for ii ∈ 1:numPts];
    push!(vConst, vξ >= 0);

    sConvProb = minimize( T(0.5) * sumsquares(paramW) + paramC * sum(vξ), vConst );
    # Hinge Loss equivalent formulation:
    # sConvProb = minimize( T(0.5) * sumsquares(paramW) + paramC * sum(pos(one(T) - vY .* (paramW * vX + paramB))) ); #<! Avoid casting with T.() in expression
    solve!(sConvProb, ECOS.Optimizer; silent = true);

    return paramW.value, paramB.value, vξ.value;
    
end

function LocateSupportVectors( vX :: Vector{T}, paramW :: T, paramB :: T ) where {T <: AbstractFloat}

    _, posSuppIdx = findmin(abs, paramW * vX .+ paramB .- one(T));
    _, negSuppIdx = findmin(abs, paramW * vX .+ paramB .+ one(T));

    return posSuppIdx, negSuppIdx;
    
end


## Parameters

vX = Float64.(collect(-3:3));
vY = [-1, -1, -1, 1, 1, 1, 1];

paramC = 1.0;

# Visualization
numGridPts = 5_000;


## Load / Generate Data

numPts = length(vX);

figureIdx += 1;

oTr = scatter(; x = vX, y = zeros(numPts), mode = "markers", 
              marker_size = 12, marker_color = vY,
              name = "Data Points", text = ["x = $(vX[ii]), y = $(vY[ii])" for ii ∈ 1:numPts]);
oLayout = Layout(title = "Data", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


## Analysis

paramC = 0.0;
paramW, paramB, vξ = SolveSoftSVM1D(vX, vY, paramC);

paramC = 1e6;
paramW, paramB, vξ = SolveSoftSVM1D(vX, vY, paramC);



## Display Analysis

# figureIdx += 1;

# oTr = heatmap(; x = vG, y = vG, z = log1p.(mO));
# oLayout = Layout(title = "Objective Function - Log Scale", width = 600, height = 600, 
#                  xaxis_title = 'x', yaxis_title = 'y',
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
# hP = Plot([oTr], oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end

