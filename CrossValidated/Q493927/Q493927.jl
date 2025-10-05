# StackExchange Cross Validated Q493927
# https://stats.stackexchange.com/questions/493927
# The Gradient of the Hinge Loss.
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
# - 1.0.000     04/10/2025  Royi Avital
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
include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function HingeLoss( vW :: Vector{T}, mS :: Matrix{T}, vY :: Vector{T} ) where {T <: AbstractFloat}
    # Hinge Loss

    dataDim    = size(mS, 1); #<! Equals to `length(vW) - 1`
    numSamples = size(mS, 2);

    vW̃ = view(vW, 1:dataDim);

    valLoss = zero(T);

    for ii in 1:numSamples
        valT = one(T) - vY[ii] * (dot(view(mS, :, ii), vW̃) + vW[end]);
        valLoss += max(zero(T), valT);
    end

    return valLoss;
    
end

function ∇HingeLoss( vW :: Vector{T}, mS :: Matrix{T}, vY :: Vector{T} ) where {T <: AbstractFloat}
    # Gradient of Hinge Loss

    dataDim    = size(mS, 1); #<! Equals to `length(vW) - 1`
    numSamples = size(mS, 2);

    vG = zeros(T, dataDim + 1);
    vT = ones(T, dataDim + 1); #<! Buffer (Extension of `mS` with one)
    vT̃ = view(vT, 1:dataDim);

    for ii in 1:numSamples
        copy!(vT̃, view(mS, :, ii));
        vG .+= ifelse(vY[ii] * dot(vT, vW) <= one(T), -vY[ii] .* vT, zero(T));
    end

    return vG;
    
end


## Parameters

# Data
numSamples = 20;
dataDim    = 3;


## Load / Generate Data

mS = randn(dataDim, numSamples); #<! Samples matrix
vY = rand((-1.0, 1.0), numSamples); #<! Labels vector

vW = randn(dataDim + 1);

## Analysis

hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = HingeLoss(vW, mS, vY);

vGRef = CalcFunGrad(vW, hObjFun);

# Numerical Gradient (Reference)



# Analytic Gradient
mX = [mS' ones(numSamples)];
mY = Diagonal(vY);

hGradF(vW :: AbstractVector{T}) where {T <: AbstractFloat} = ∇HingeLoss(vW, mS, vY); #<! Analytic by loop
# hGradF(vW :: AbstractVector{T}) where {T <: AbstractFloat} = sum(-(mX' * mY) .* reshape((mY * mX * vW) .< one(T), (1, :)); dims = 2); #<! Analytic vectorized

vG = hGradF(vW);


# Verify Analysis

norm(vG - vGRef, Inf)