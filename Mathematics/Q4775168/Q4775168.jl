# StackExchange Mathematics Q4775168
# https://math.stackexchange.com/questions/4775168
# Optimization of the Sum of a Quadratic Form and the L1 Norm of the Logarithm.
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
# - 1.0.000     11/08/2025  Royi Avital
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
# include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( vX :: Vector{T}, mW :: Matrix{T}, λ :: T ) where {T <: AbstractFloat}

    vT = log.(vX);

    return dot(vX, mW, vX) + λ * sum(abs, vT);
    
end


function ProxObjFun( vX :: Vector{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}

    vT     = vX - vY;
    objVal = T(0.5) * sum(abs2, vT);

    vT      = log.(vX);
    objVal += λ * sum(abs, vT);

    return objVal;
    
end

function ProxNorm1Log( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}

    numElements = length(vY);
    vX          = zero(vY);

    for ii in 1:numElements
        if abs(vY[ii]) <= T(1e-8)
            vX[ii] = zero(T);
            continue;
        end
        hF(x :: T) = (x - vY[ii]) + λ * sign(x) / x;
        vX[ii] = FindZeroBinarySearch(hF, T(1e-6), T(1000));
    end

    return vX;
    
end


## Parameters

# Data
numElements = 10;

vValY      = [-2.0, -0.1, 0.1, 1.0, 2.0];
numGridPts = 1_000;
λ          = 0.75;

# Sub Gradient Function per Element
hSubGradI( x :: T, y :: T, λ :: T ) where {T <: AbstractFloat} = x - y + λ * sign(log(x)) / x;


## Load / Generate Data

vValX = collect(LinRange(0.1, 5, numGridPts));
vValZ = zeros(numGridPts);

vY = rand(numElements);

## Analysis


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(vValY));

for (ii, valY) in enumerate(vValY)
    for (jj, valX) in enumerate(vValX)
        vValZ[jj] = hSubGradI(valX, valY, λ);
    end
    lineName = @sprintf("y = %0.2f", valY);
    vTr[ii] = scatter(x = vValX, y = copy(vValZ), mode = "lines",
                      line = attr(width = 3.0), text = lineName, name = lineName)
end

sLayout = Layout(title = "The Element of the Sub Gradient", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "Sub Gradient Value",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

