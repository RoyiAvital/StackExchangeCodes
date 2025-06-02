# StackExchange Mathematics Q2143044
# https://math.stackexchange.com/questions/2143044
# Optimizing the Hyper Parameter of Huber Loss.
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
# - 1.0.000     02/06/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using SCS;
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

function GenDataSamples( tuInModel :: Tuple{T, T}, tuOutModel :: Tuple{T, T}, vX :: Vector{T}, numOutliers :: N, σ :: T, oRng :: oRNG ) where {T <: AbstractFloat, N <: Integer, oRNG <: AbstractRNG}

    numSamples = length(vX);
    
    vY = tuInModel[1] * vX .+ tuInModel[2];   #<! Inlier samples
    vO = tuOutModel[1] * vX .+ tuOutModel[2]; #<! Outlier samples
    vN = σ .* randn(oRng, T, numSamples);           #<! Noise samples

    # Outlier indices
    vI = rand(oRng, 1:numSamples, numOutliers);

    vY[vI] .= vO[vI];
    vY    .+= vN;

    return vY, vI;

end

function SolveHuberReg( mX :: Matrix{T}, vY :: Vector{T}, δ :: T ) where {T <: AbstractFloat}

    numVar = size(mX, 2);
    vΘ     = Variable(numVar);    #<! Regression parameters
    
    sConvProb = minimize( 0.5 * sum(huber(mX * vΘ - vY, δ)) );
    solve!(sConvProb, SCS.Optimizer; silent = true);

    return vec(vΘ.value); #<! Return a vector

end

## Parameters

# Data
numSamples   = 50;
numOutliers  = 10; #<! Examples
vNumOutliers = 0:20; #<! Analysis

# Model
tuGrid = (0.0, 2.0);

# Inliers Model
tuInModel  = (0.65, -0.70); #<! (a, b)
# Outliers Model
tuOutModel = (-0.70, 0.90); #<! (a, b)

σ = 0.05; #<! Noise Level

# Huber Model
δ       = 0.2; #<! Example
vParamδ = collect(LinRange(0.0, 1.5, 21)); #<! Analysis

#%% Load / Generate Data

# Grid
vX = collect(LinRange(tuGrid[1], tuGrid[2], numSamples));
mX = [vX;; vX .^ 0]; #<! Model Matrix

## Analysis

# The 2 Models
vY1 = tuInModel[1] .* vX .+ tuInModel[2];
vY2 = tuOutModel[1] .* vX .+ tuOutModel[2];

# Example with 10 Outliers
vY, vI   = GenDataSamples(tuInModel, tuOutModel, vX, numOutliers, σ, oRng);

# Least Squares Model
vLsModel = mX \ vY;
vYLs     = mX * vLsModel;

# Huber Model
vHuberModel = SolveHuberReg(mX, vY, δ);
vYHuber     = mX * vHuberModel;

# Sensitivity to δ
mS = zeros(length(vParamδ), length(vNumOutliers));
vZ = mX * collect(tuInModel); #<! Ground Truth

for (jj, numOutliers) ∈ enumerate(vNumOutliers)
    vYY, _ = GenDataSamples(tuInModel, tuOutModel, vX, numOutliers, σ, oRng);
    for (ii, δδ) ∈ enumerate(vParamδ)
        if δδ <= eps(Float64)
            # Huber with δ = 0 -> LS
            vModelE = mX \ vYY; #<! Estimated Model
        else
            vModelE = SolveHuberReg(mX, vYY, δδ);
        end

        vYYE       = mX * vModelE; #<! Estimation
        mS[ii, jj] = sqrt(sum(abs2, vZ - vYYE)); #<! RMSE
    end
end

## Display Results

figureIdx += 1;

sTr1 = scatter(x = vX, y = vY1, mode = "lines", name = "Inlier Model");
sTr2 = scatter(x = vX, y = vY2, mode = "lines", name = "Outlier Model");

sLayout = Layout(title = "Data Models", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "x", yaxis_title = "y", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975));

vTr = [sTr1, sTr2];
hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end

figureIdx += 1;

sTr1 = scatter(x = vX, y = vY, mode = "markers", name = "Samples ($(numOutliers) Outliers)");
sTr2 = scatter(x = vX[vI], y = vY[vI], mode = "markers", marker_size = 10, marker_opacity = 0.35, name = "Outlier Model Samples");
sTr3 = scatter(x = vX, y = vYLs, mode = "lines", line_width = 4, name = "LS Estimation");
sTr4 = scatter(x = vX, y = vYHuber, mode = "lines", line_width = 4, name = "Huber Estimation δ = $(δ)");

sLayout = Layout(title = "Data Models", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "x", yaxis_title = "y", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975));

vTr = [sTr1, sTr2, sTr3, sTr4];
hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end

figureIdx += 1;

sTr = heatmap(x = [@sprintf("%d", numOutliers) for numOutliers ∈ vNumOutliers], y = [@sprintf("%0.2f", δ) for δ ∈ vParamδ], z = mS);

sLayout = Layout(title = "RMSE of the Estimation (Ground Truth)", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "Number of Outliers", yaxis_title = "δ", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975));

vTr = [sTr];
hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end
