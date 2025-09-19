# StackExchange Mathematics Q1652661
# https://math.stackexchange.com/questions/1652661
# Optimizing the Binary Logistic Regression with ${L}_{2}$ Squared Regularization Term.
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
# - 1.0.000     19/09/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
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
include(joinpath(juliaCodePath, "JuliaSciFun.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( mX :: Matrix{T}, vY :: Vector{T}, vW :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # `mX` is (dataDim, numSamples)
    # `vW` is (dataDim + 1,) (Weights + Intercept)
    # The probability: pᵢ = P(yᵢ = 1 | xᵢ) = σ( zᵢ ), zᵢ = \frac{1}{1 + exp( -(xᵀ w + b) )}
    # The Log Loss: -∑ yᵢ log (pᵢ) + (1 - yᵢ) log(1 - pᵢ) = ∑ log(1 + exp(xᵀ w + b) - ∑ yᵢ (xᵀ w + b)
    # The Objective: LogLoss(w) + 0.5 * λ || w ||_2^2

    dataDim    = size(mX, 1);
    numSamples = size(mX, 2);

    valLoss = zero(T);

    for ii in 1:numSamples
        zᵢ = dot(view(mX, :, ii), view(vW, 1:dataDim)) + vW[end]; #<! Linear Model
        # pᵢ = Logistic(zᵢ); #<! Sigmoid
        # valLoss += -vY[ii] * log(pᵢ) - (one(T) - vY[ii]) * log(one(T) - pᵢ); #<! -(Log Likelihood)
        valLoss += log1p(exp(zᵢ)) - vY[ii] * zᵢ; #<! Simplified
    end

    valLoss += T(0.5) * λ * sum(abs2, vW); #<! Regularization
    
    return valLoss;
    
end

function ∇ObjFun( mX :: Matrix{T}, vY :: Vector{T}, vW :: Vector{T}, λ :: T ) where {T <: AbstractFloat}

    dataDim    = size(mX, 1);
    numSamples = size(mX, 2);

    v∇W = zeros(T, dataDim + 1);
    v∇W₀ = view(v∇W, 1:dataDim); #<! Weights
    v∇W₁ = view(v∇W, dataDim + 1); #<! Intercept

    for ii in 1:numSamples
        zᵢ = dot(view(mX, :, ii), view(vW, 1:dataDim)) + vW[end]; #<! Linear Model
        pᵢ = Logistic(zᵢ); #<! Sigmoid
        v∇W₀ .+= (pᵢ - vY[ii]) .* view(mX, :, ii);
        v∇W₁ .+= (pᵢ - vY[ii]);
    end

    v∇W .+= λ .* vW; #<! Regularization

    return v∇W;
    
end

function Predict( vW :: Vector{T}, vX :: Vector{T} ) where {T <: AbstractFloat}

    return (vW[1] * vX[1] + vW[2] * vX[2] + vW[3]) > zero(T);

end

function Predict( vW :: Vector{T}, x1 :: T, x2 :: T ) where {T <: AbstractFloat}

    return (vW[1] * x1 + vW[2] * x2 + vW[3]) > zero(T);

end


## Parameters

# Data
fileName = "BinaryClassifierData.csv";
dataDim = 2;

# Model
λ = 0.25; #<! Has no effect on this dataset

# Solver
numIter = 10_000;
η       = 1e-5; #<! Step Size


## Load / Generate Data

mD = readdlm(fileName, ',', Float64, skipstart = 1);
mX = collect(mD[:, 1:2]');
vY = mD[:, 3];

numSamples = size(mX, 2);

## Analysis

hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = ObjFun(mX, vY, vW, λ);
h∇Fun(vW :: Vector{T}) where {T <: AbstractFloat} = ∇ObjFun(mX, vY, vW, λ);

# Finite Differences solution
vW = 0.75 * randn(dataDim + 1);
vGRef = CalcFunGrad(vW, hObjFun; ε = 1e-6);
vG = h∇Fun(vW);

maximum(abs.(vG - vGRef))

vW = zero(vW);
vW = GradientDescent(vW, numIter, η, h∇Fun);

numSucc = 0;
for ii in 1:numSamples
    global numSucc;
    numSucc += vY[ii] == Predict(vW, mX[:, ii]);
end
valAcc = numSucc / numSamples;


## Display Results

numGridPts = 2_001;
vXX = collect(LinRange(-2, 2, numGridPts));
mZ  = zeros(numGridPts, numGridPts);

for (jj, x1) in enumerate(vXX)
    for (ii, x2) in enumerate(vXX)
        mZ[ii, jj] = Predict(vW, x1, x2);
    end
end

figureIdx += 1;

vL  = sort(unique(vY));
vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(vL) + 1);

for (ii, valL) in enumerate(vL)
    clsStr = @sprintf("%d", valL);
    vIdx = vY .== valL;
    vTr[ii] = scatter(x = mX[1, vIdx], y = mX[2, vIdx], mode = "markers",
                      marker = attr(size = 12, color = vPlotlyDefColors[ii]), text = clsStr, name = clsStr)
end

vTr[3] = contour(; x = vXX, y = vXX, z = mZ,
                 ncontours = 1,
                 autocontour = false,
                 colorscale = [[0, vPlotlyDefColors[1]], [0.5, vPlotlyDefColors[1]], [0.5, vPlotlyDefColors[2]], [1, vPlotlyDefColors[2]]],
                 contours_start = 0, contours_end = 1, contours_size = 1,
                 contours_coloring = "heatmap",
                 opacity = 0.50);

titleStr =  @sprintf("Binary Logistics Regression Classifier, Accuracy: %0.2f%%", 100.0 * valAcc);

sLayout = Layout(title = titleStr, width = 600, height = 600, 
                 xaxis_title = "x₁", yaxis_title = "x₂",
                 xaxis_range = (-2.05, 2.05), yaxis_range = (-2.05, 2.05),
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

