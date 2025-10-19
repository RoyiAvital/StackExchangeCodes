# StackExchange Cross Validated Q215733
# https://stats.stackexchange.com/questions/215733
# Use Sub Gradient to Solve the Primal (Kernel) SVM (Pegasos Algorithm).
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
# - 1.0.000     17/10/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
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

function SVM( vW :: Vector{T}, mX :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}
    # The Kernel SVM without the bias term

    numSamples = length(vY);
    
    # Regularization
    objVal = 0.5 * λ * dot(vW, vW);

    # Objective
    vH = [max(zero(T), T(1) - vY[ii] * dot(vW, mX[:, ii])) for ii in 1:numSamples];
    if squareHinge
        vH = [vH[ii] * vH[ii] for ii in 1:numSamples];
    end

    objVal += sum(vH) / numSamples;

    return objVal;

end

function SVMCVX( mX :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # The SVM without the bias term

    # `mX` the data samples matrix
    dataDim     = size(mX, 1);
    numSamples  = size(mX, 2);

    vW = Convex.Variable(dataDim);

    hingeLoss = Convex.sum(Convex.pos(T(1) - vY .* (mX' * vW)));
    hingeLoss = hingeLoss / numSamples; #<! Stochastic sample only one of the corpus

    sConvProb = minimize( T(0.5) * λ * Convex.sumsquares(vW) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vW.value);
    
end

function PegasosSVM!( mW :: Matrix{T}, hXₖ! :: Function, vY :: Vector{T}, λ :: T, vWₜ :: Vector{T}, vθₜ :: Vector{T}, vXₜ :: Vector{T} ) where {T <: AbstractFloat}
    # Following Shai Shalev Shwartz, Shai Ben David - Understanding Machine Learning: From Theory to Algorithms.
    # See page 213 - SGD for Solving Soft SVM.
    # `hXₖ!(vZ, ii)` - Returns the `ii` -th data sample in `vZ`.

    numSamples    = length(vY);
    dataDim       = size(mW, 1);
    numIterations = size(mW, 2);

    # First iteration
    tt = 1;
    ηₜ = inv(λ * tt);

    vWₜ .= ηₜ .* vθₜ;

    kk = rand(1:numSamples);
    yₖ = vY[kk];

    hXₖ!(vXₜ, kk);

    valSum = yₖ * dot(vWₜ, vXₜ);
    if valSum < one(T)
        vθₜ .+= yₖ .* vXₜ;
    end

    @views mW[:, 1] = vWₜ;


    for ii in 2:numIterations

        # tt = ii - 1;
        tt = ii;
        ηₜ = inv(λ * tt);

        vWₜ .= ηₜ .* vθₜ;

        kk = rand(1:numSamples);
        yₖ = vY[kk];

        hXₖ!(vXₜ, kk);

        valSum = yₖ * dot(vWₜ, vXₜ);
        if valSum < one(T)
            vθₜ .+= yₖ .* vXₜ;
        end

        @views mW[:, ii] .= inv(tt) .* ((T(ii - 1) .* mW[:, ii - 1]) .+ vWₜ);
    end

    return mW;

end

function PredictSVM( vW :: Vector{T}, vX :: Vector{T} ) where {T <: AbstractFloat}

    return sign(dot(vW, vX));

end


## Parameters

# Data
csvFileName = raw"BinaryClassificationData.csv";

# SVM Model
λ = 0.1;

# Solvers
numIterations = 2_000;

# Visualization
numGridPts = 2_001;

## Load / Generate Data

mD = readdlm(csvFileName, ',', Float64; skipstart = 1);
mX = collect(mD[:, 1:2]'); #<! Sample in a row -> Sample in a column
vY = mD[:, 3];
vY .= 2.0 .* vY .- 1.0; #<! Map {0, 1} -> {-1, 1}

# Shift to Center
mX[1, :] .-= mean(mX[1, :]);
mX[2, :] .-= mean(mX[2, :]);

dataDim    = size(mX, 1);
numSamples = size(mX, 2);

hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = SVM(vW, mX, vY, λ);

dSolvers = Dict();

## Analysis

# DCP Solver
methodName = "Convex.jl"

vWRef  = SVMCVX(mX, vY, λ);
optVal = hObjFun(vWRef);

dSolvers[methodName] = optVal * ones(numIterations);

# Pegasos Method
methodName = "Pegasos";

vWₜ = zeros(dataDim);
vθₜ = zeros(dataDim);
vXₜ = zeros(dataDim);

hXₖ!( vX :: Vector{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer} = copy!(vX, view(mX, :, ii));

mW = zeros(dataDim, numIterations);

mW = PegasosSVM!(mW, hXₖ!, vY, λ, vWₜ, vθₜ, vXₜ);

dSolvers[methodName] = [hObjFun(mW[:, ii]) for ii ∈ 1:size(mW, 2)];

vW = mW[:, end];
vŶ = [PredictSVM(vW, mX[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);

vXX = collect(LinRange(-2, 2, numGridPts));
mZ  = zeros(numGridPts, numGridPts);

for (jj, x1) in enumerate(vXX)
    for (ii, x2) in enumerate(vXX)
        vXₜ[1] = x1;
        vXₜ[2] = x2;
        mZ[ii, jj] = PredictSVM(vW, vXₜ);
    end
end


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]",
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value",
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
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
                 colorscale = [[0, vPlotlyDefColors[1]], [0.5, vPlotlyDefColors[1]], [0.5, vPlotlyDefColors[2]], [1, vPlotlyDefColors[2]]], #<! On the [0, 1] range
                 contours_start = -1.0, contours_end = 1.0, contours_size = 1,
                 contours_coloring = "heatmap",
                 opacity = 0.50);

titleStr =  @sprintf("SVM Classifier, Accuracy: %0.2f%%", 100.0 * valAcc);

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

