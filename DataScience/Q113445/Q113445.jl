# StackExchange Data Science Q113445
# https://datascience.stackexchange.com/questions/113445
# 2D Localization Using Weighted Non Linear Least Squares.
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
# - 1.0.000     22/03/2026  Royi Avital
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


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function _LevenbergMarquardt( vβ :: Vector{T}, λ :: T, vY :: Vector{T}, mX :: Matrix{T}, hF :: Function, h∇f :: Function; vW :: Vector{T} = ones(T, length(vY)), λFactor :: T = T(10) ) where {T <: AbstractFloat}
    # Single iteration of a stateless implementation of Levenberg Marquardt algorithm.

    numSamples = size(mX, 2);
    numParams  = length(vβ);

    length(vY) == numSamples || error("The number of samples in `mX` must match the length of `vY`");
    length(vW) == numSamples || error("The length of `vW` must match the length of `vY`");
    λ > zero(T) || error("The damping factor `λ` must be positive");
    λFactor > one(T) || error("`λFactor` must be larger than 1");

    vβ      = copy(vβ);
    vβCand  = similar(vβ);
    vΔβ     = similar(vβ);
    vG      = similar(vβ);
    vR      = Vector{T}(undef, numSamples);
    vRCand  = similar(vR);
    mJ      = Matrix{T}(undef, numSamples, numParams);
    mA      = Matrix{T}(undef, numParams, numParams);

    function CalcObjJac!( vR :: Vector{T}, mJ :: Matrix{T}, vY :: Vector{T}, mX :: Matrix{T}, vβ :: Vector{T}, vW :: Vector{T} ) where {T <: AbstractFloat}
        objVal = zero(T);

        for ii ∈ 1:numSamples
            vXi = view(mX, :, ii);
            valW = vW[ii];

            sqrtW = sqrt(valW);
            valRes = vY[ii] - hF(vXi, vβ);
            vJ = h∇f(vXi, vβ);

            vR[ii] = sqrtW * valRes;
            @views mJ[ii, :] .= sqrtW .* vJ;
            objVal += abs2(vR[ii]);
        end

        return objVal;
    end

    function CalcObj( vR :: Vector{T}, vY :: Vector{T}, mX :: Matrix{T}, vβ :: Vector{T}, vW :: Vector{T} ) where {T <: AbstractFloat}
        objVal = zero(T);

        for ii ∈ 1:numSamples
            vXi = view(mX, :, ii);
            valW = vW[ii];

            vR[ii] = sqrt(valW) * (vY[ii] - hF(vXi, vβ));
            objVal += abs2(vR[ii]);
        end

        return objVal;
    end

    objVal = CalcObjJac!(vR, mJ, vY, mX, vβ, vW);

    mul!(mA, mJ', mJ);
    mul!(vG, mJ', vR);

    for ii ∈ 1:numParams
        mA[ii, ii] += λ;
    end

    vΔβ    .= mA \ vG;
    vβCand .= vβ .+ vΔβ;

    if !all(isfinite, vβCand)
        return vβ, λ * λFactor;
    end

    objValCand = CalcObj(vRCand, vY, mX, vβCand, vW);

    if objValCand <= objVal
        vβ .= vβCand;
        return vβ, max(λ / λFactor, eps(T));
    end

    return vβ, λ * λFactor;

end


## Parameters

# Data
tuGridSize = (10, 10); #<! (x, y) with origin at (0, 0)

# Base Stations (Anchors)
mX = [1.0 1.0; 10.0 5.0; 2.0 4.0];
mX = Matrix(mX'); #<! Each column is a point
vV = [0.35, 0.55, 0.75]; #<! Variance of the noise for each anchor

# Solver / Solution
numGridPts = 1_000; #<! Per dimension

# Solver
numIter = 1_000;
η       = 1e-3;

# Visualization


## Load / Generate Data

numAnchors = size(mX, 2);

vPtLocation = [rand(oRng) * tuGridSize[1], rand(oRng) * tuGridSize[2]];
vMeasNoise  = sqrt.(vV) .* randn(oRng, numAnchors);
vDist       = [sum(abs2, mX[:, ii] - vPtLocation) for ii ∈ 1:numAnchors];
vY          = vMeasNoise + vDist;


## Analysis

vβ₀ = [5.0, 5.0]; #<! Initial guess for the point location
vW  = 1.0 ./ vV; #<! Weighting vector for the anchors
vW  = ones(numAnchors);

# Squared Euclidean Distance
hF( vX :: AbstractVector{T}, vβ :: AbstractVector{T} ) where {T <: AbstractFloat} = sum(abs2, vβ - vX);
# Gradient
h∇f( vX :: AbstractVector{T}, vβ :: AbstractVector{T} ) where {T <: AbstractFloat} = T(2) * (vβ - vX);
# Objective Function
hObjFun( vβ :: AbstractVector{T} ) where {T <: AbstractFloat} = sum(sum(abs2, hF(mX[:, ii], vβ) - vY[ii]) for ii ∈ 1:numAnchors);

# Grid Search 
vGx = LinRange(0, tuGridSize[1], numGridPts);
vGy = LinRange(0, tuGridSize[2], numGridPts);

mG = [hObjFun([xx, yy]) for xx in vGx, yy in vGy];
vMinIdx = argmin(mG);
vP = [vGx[vMinIdx[1]], vGy[vMinIdx[2]]];

# Levenberg Marquardt
vβ = LevenbergMarquardt(vY, mX, vβ₀, hF, h∇f; vW = vW, λFactor = 1.5);

mβ = zeros(length(vβ₀), 10);
mβ[:, 1] = vβ₀;
λ = 1e-3;

for ii = 2:size(mβ, 2)
    global λ;
    mβ[:, ii], λ = _LevenbergMarquardt(mβ[:, ii - 1], λ, vY, mX, hF, h∇f; vW = vW, λFactor = 1.5);
end

vMeasRadius = sqrt.(max.(vY, zero(eltype(vY))));


## Display Analysis

figureIdx += 1;

oTr1 = scatter(; x = mX[1, :], y = mX[2, :], mode = "markers", 
                marker_size = 12, text = "Access Point", name = "Access Points");
oTr2 = scatter(; x = [vPtLocation[1]], y = [vPtLocation[2]], mode = "markers", 
                marker_size = 12, text = "Reference Point", name = "Reference Point");
oShp = [circle(x0 = mX[1, ii] - vMeasRadius[ii], y0 = mX[2, ii] - vMeasRadius[ii], 
                x1 = mX[1, ii] + vMeasRadius[ii], y1 = mX[2, ii] + vMeasRadius[ii];
                opacity = 0.15, fillcolor = "black", line_color = "white") for ii ∈ 1:numAnchors];
oLayout = Layout(title = "Localization by Range Measurements: Scenario", width = 600, height = 600, 
                xaxis_range = [0, tuGridSize[1]], yaxis_range = [0, tuGridSize[2]], xaxis_title = 'x', yaxis_title = 'y',
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                legend = attr(x = 0.025, y = 0.975), shapes = oShp);
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme, width = hP.layout["width"], height = hP.layout["height"]);
end

figureIdx += 1;

oTr1 = heatmap(; x = vGx, y = vGy, z = log1p.(abs.(mG)));
oTr2 = scatter(; x = mX[1, :], y = mX[2, :], mode = "markers", 
                marker_size = 12, text = "Access Point", name = "Access Points");
oTr3 = scatter(; x = [vPtLocation[1]], y = [vPtLocation[2]], mode = "markers", 
                marker_size = 12, text = "Reference Point", name = "Reference Point");
oTr4 = scatter(; x = [vP[1]], y = [vP[2]], mode = "markers", 
                marker_size = 12, text = "Estimated Point", name = "Estimated Point");
oTr5 = scatter(; x = mβ[1, :], y = mβ[2, :], mode = "lines+markers", 
                marker_size = 7, line_width = 2, text = "Optimizer Path", name = "Optimizer Path");
oShp = [circle(x0 = mX[1, ii] - vMeasRadius[ii], y0 = mX[2, ii] - vMeasRadius[ii], 
                x1 = mX[1, ii] + vMeasRadius[ii], y1 = mX[2, ii] + vMeasRadius[ii];
                opacity = 0.15, fillcolor = "black", line_color = "white") for ii ∈ 1:numAnchors]
oLayout = Layout(title = "Localization by Range Measurements: Scenario", width = 600, height = 600, 
                xaxis_range = [0, tuGridSize[1]], yaxis_range = [0, tuGridSize[2]], xaxis_title = 'x', yaxis_title = 'y',
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                legend = attr(x = 0.025, y = 0.975), shapes = oShp);
hP = Plot([oTr1, oTr2, oTr3, oTr4, oTr5], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme, width = hP.layout["width"], height = hP.layout["height"]);
end

