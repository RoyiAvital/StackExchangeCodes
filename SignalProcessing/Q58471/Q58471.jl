# StackExchange Signal Processing Q58471
# https://dsp.stackexchange.com/questions/58471
# Minimize the Maximum Value of a Sum of Cosine by Optimizing the Phase of Each Component.
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
# - 1.0.000     13/02/2025  Royi Avital
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
using StaticKernels;       #<! Required for Image / Signal Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl")); #<! Optimization
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ProjectPhase!( vI :: Vector{T} ) where {T <: AbstractFloat}

    numElements = length(vI);
    for ii ∈ 1:2:numElements
        vJ   = view(vI, ii:(ii + 1));
        vJ ./= norm(vJ);
    end

    return vI;

end

function CalcObjVal( mX :: Matrix{T}, vα :: Vector{T} ) where {T <: AbstractFloat}

    vY = mX * vα;

    return maximum(abs.(vY));

end

function CalcObjGrad( mX :: Matrix{T}, vα :: Vector{T} ) where {T <: AbstractFloat}
    # Gradient of || X * α ||_∞
    # Efficient calculation of: X' * sign((X * α)_j) where j = \arg \max_i (|X * α|)_i

    vT        = mX * vα;
    _, maxIdx = findmax(abs, vT);
    maxVal    = vT[maxIdx];
    vG        = sign(maxVal) .* mX[maxIdx, :];

    return vG;

end

## Parameters

# Problem parameters
numSamples = 1_001;
numSignals = 10;

ε = 1e-6;

# Solver Parameters
numTrials = 5_000_000;
η         = 1e-4; #<! Step size
numIter   = 1_000;

## Load / Generate Data

vX = LinRange(-2, 2, numSamples);
vF = LinRange(1.0, numSignals, numSignals);
mX = [cos.(2π .* vX .* vF'); sin.(2π .* vX .* vF')];
mX = reshape(mX, (numSamples, 2 * numSignals));

vα = zeros(2 * numSignals);
vT = zeros(2 * numSignals);

## Analysis

# Random Initialization
minAmp = 10.0;
for ii ∈ 1:numTrials
    vT[:] = (2 .* rand(oRng, 2 * numSignals)) .- 1;
    ProjectPhase!(vT);
    currAmp = CalcObjVal(mX, vT);
    if currAmp < minAmp
        global minAmp = currAmp;
        copy!(vα, vT);
    end
end

println("The minimum amplitude is given by: $(minAmp)")

# Verify the Gradient Function

hCalcObj(vX) = CalcObjVal(mX, vX);
vG = CalcFunGrad(vα, hCalcObj);
vT = CalcObjGrad(mX, vα);

@assert (norm(vG - vT, Inf) < ε) "The gradient is not verified";

# Optimize Using Projected Gradient Descent

hGradFun(vX)    = CalcObjGrad(mX, vX);
hGradFun(vX)    = CalcFunGrad(vX, hCalcObj);
hProxFun(vX, λ) = ProjectPhase!(vX);


vT = ProximalGradientDescent(vα, hGradFun, hProxFun, η, numIter);
println("The minimum amplitude is given by: $(CalcObjVal(mX, vT))");

## Display Results

# Display Data
figureIdx += 1;

oTr = scatter(x = collect(vX), y = mX * vα, 
              mode = "lines", text = "Signal", name = "Signal", 
              line = attr(width = 3.0));
oLayout = Layout(title = "Combined Signal", width = 600, height = 600, 
                 hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot(oTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
