# StackExchange Cross Validated Q23037
# https://stats.stackexchange.com/questions/23037
# Efficient Solver of the Kernel SVM for Large Scale Data.
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
# - 1.0.000     18/10/2025  Royi Avital
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
using StatsBase;


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

function SVM( vW :: Vector{T}, mX :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # The Kernel SVM without the bias term

    numSamples = length(vY);
    
    # Regularization
    objVal = 0.5 * λ * dot(vW, vW);

    # Objective
    vH = [max(zero(T), T(1) - vY[ii] * dot(vW, mX[:, ii])) for ii in 1:numSamples];

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

function CalcRbfKernel( vX₁ :: AbstractVector{T}, vX₂ :: AbstractVector{T}, γ :: T ) where {T <: AbstractFloat}

    numSamples = length(vX₁);

    sumDiff = zero(T);
    for ii in 1:numSamples
        valDiff  = vX₁[ii] - vX₂[ii];
        sumDiff += valDiff * valDiff;
    end

    return exp(-γ * sumDiff);

end


function CalcRbfKernelMat( mX :: Matrix{T}, σ :: T; α :: T = zero(T) ) where {T <: AbstractFloat}
    # Calculates the RBF Kernel Matrix
    
    dataDim = size(mX, 1);
    numSamples = size(mX, 2);
    mK = ones(T, numSamples, numSamples);

    γ = inv(T(2) * σ * σ);

    for jj in 2:numSamples, ii in 1:(jj - 1)
        mK[ii, jj]  = CalcRbfKernel(view(mX, :, ii), view(mX, :, jj), γ);
        mK[jj, ii]  = mK[ii, jj];
    end

    mK .+= α .* I(numSamples);

    return mK;

end

function PegasosSVM!( mW :: Matrix{T}, hXᵢ! :: Function, vY :: Vector{T}, λ :: T, numIter :: N ) where {T <: AbstractFloat, N <: Integer}
    # Following Shai Shalev Shwartz, Shai Ben David - Understanding Machine Learning: From Theory to Algorithms.
    # See page 213 - SGD for Solving Soft SVM.
    # `hXₖ!(vZ, ii)` - Returns the `ii` -th data sample in `vZ`.

    numSamples  = length(vY);
    numFeatures = size(mW, 1);
    numIter     = size(mW, 2) - 1;

    vW = zeros(T, numFeatures);
    vWₜ = zeros(T, numFeatures);
    vθₜ = zeros(T, numFeatures);
    vXₜ = zeros(T, numFeatures);

    jj = 2;
    for tt in 1:numIter

        ηₜ = inv(λ * tt);

        vWₜ .= ηₜ .* vθₜ;

        ii = rand(1:numSamples);
        yᵢ = vY[ii];

        hXᵢ!(vXₜ, ii);

        valSum = yᵢ * dot(vWₜ, vXₜ);
        if valSum < one(T)
            vθₜ .+= yᵢ .* vXₜ;
        end

        vW .+= vWₜ;
        @views mW[:, jj] .= vW ./ tt;
        jj += 1;
    end

    return mW;

end

function ProjectL2Ball!( vX :: AbstractVector{T}, vY :: AbstractVector{T}, vC :: AbstractVector{T}, ballRadius :: T ) where {T <: AbstractFloat}
    # Projects `vY` to the radius with center `vC` and radius `ballRadius`.
    # The output is in `vX`.
    
    vX .= vY .- vC;
    valDist = norm(vX);

    if valDist > ballRadius
        projFactor = (ballRadius / valDist);
        vX .= vC .+ projFactor .* vX;
    else
        copy!(vX, vY);
    end

    return vX;

end

function SnacksSVM!( mW :: Matrix{T}, hXᵢ! :: Function, vY :: Vector{T}, λ :: T, numRestarts :: N, ω :: T, η :: T, ballRadius :: T ) where {T <: AbstractFloat, N <: Integer}
    # Snacks: A Fast Large Scale Kernel SVM Solver (https://arxiv.org/abs/2304.07983)

    numSamples  = length(vY);
    numFeatures = size(mW, 1);
    numIter     = size(mW, 2) - 1;

    numInIter = numIter ÷ numRestarts;

    vW  = zeros(T, numFeatures);
    vC  = zeros(T, numFeatures);
    vW̄  = zeros(T, numFeatures);
    vKᵢ = zeros(T, numFeatures);
    vG  = zeros(T, numFeatures);
    vB  = zeros(T, numFeatures);

    jj = 2;
    for kk in 1:numRestarts
        copy!(vC, vW);
        # copy!(vW̄, vW); #<! The paper
        vW̄ .= zero(T); #<! The code (See https://github.com/SofianeTanji/snacks/issues/2)

        for tt in 1:numInIter
            ii = rand(1:numSamples);
            yᵢ = vY[ii];
            vKᵢ = hXᵢ!(vKᵢ, ii);
            valFlag  = (yᵢ * dot(vW, vKᵢ)) < one(T);
            @. vG = λ * vW - valFlag * yᵢ * vKᵢ;
            vW .-= η * vG;
            vB = ProjectL2Ball!(vB, vW, vC, ballRadius);
            vW̄ .+= vB ./ T(numInIter);
            copy!(view(mW, :, jj), vW);
            jj += 1; 
        end
        copy!(vW, vW̄);
        ballRadius /= ω;
        η /= ω;
    end

    return mW;

end

function ApproxEmbedder( mC :: Matrix{T}, σ :: T; α :: T = zero(T) ) where {T <: AbstractFloat}
    # Generates approximates embedding by `mC` centers (Number of features).
    # Embedding new dataset `mX` by:
    # 1. Calculating `mK(i, j) = RBF(mX[:, i], mC[:, j])`.
    # 2. `mKEmb = mK / sKc`;
    # 3. Normalize `mKEmb` (Zero mean, Unit variance).

    mKc = CalcRbfKernelMat(mC, σ; α = α);
    sKc = cholesky(mKc);

    return mKc, sKc;

end

function EmbedData( mX :: Matrix{T}, mC :: Matrix{T}, sKc, σ :: T ) where {T <: AbstractFloat}
    # Embeds data by `mC` centers and `sKc` which is the decomposition (Bunch Kaufman) of `mKc`.

    numSamples  = size(mX, 2);
    numFeatures = size(mC, 2);

    γ = inv(T(2) * σ * σ);

    mK = zeros(T, numFeatures, numSamples);

    for jj in 1:numSamples, ii in 1:numFeatures
        mK[ii, jj] = CalcRbfKernel(view(mC, :, ii), view(mX, :, jj), γ);
    end
    
    mKe = sKc \ mK;

    return mKe;

end

function PredictSVM( vW :: Vector{T}, vX :: Vector{T} ) where {T <: AbstractFloat}

    return sign(dot(vW, vX));

end


## Parameters

# Data
csvFileName = raw"BinaryClassificationData.csv"; #<! Binary Data

# Embedding
numCenters = 50;

# SVM Model
σ = 0.025;
λ = 0.1;
α = 1e-5; #<! Regularization to make `mK` SPD

# Solvers
# Pegasos
numIterations = 5_000;

# Snacks
numRestarts = 10; #<! K
ballRadius  = 10.0;
ω           = 2.0; #<! Shrinking factor
η           = 5e-1; #<! Initial step size

# Visualization
numGridPts = 2_001;

## Load / Generate Data

mD = readdlm(csvFileName, ',', Float64; skipstart = 1);
mX = collect(mD[:, 1:2]'); #<! Sample in a row -> Sample in a column
vY = mD[:, end];
vY .= 2.0 .* vY .- 1.0; #<! Map {0, 1} -> {-1, 1}

# Shift to Center
mX .-= mean(mX; dims = 2);

dataDim    = size(mX, 1);
numSamples = size(mX, 2);

vCenIdx = sample(oRng, 1:numSamples, numCenters; replace = false);
mC      = mX[:, vCenIdx];
numFeat = numCenters;

# Create approximated embeddings
# Once the embeddings are available, the problem can be solved using LinearSVM
mKc, sKc = ApproxEmbedder(mC, σ; α = α);
mK = EmbedData(mX, mC, sKc, σ);
hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = SVM(vW, mK, vY, λ);

dSolvers = Dict();

## Analysis

# DCP Solver
methodName = "Convex.jl"

vWRef  = SVMCVX(mK, vY, λ);
optVal = hObjFun(vWRef);

dSolvers[methodName] = optVal * ones(numIterations);

vŶ = [PredictSVM(vWRef, mK[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);
println(@sprintf("%s: Accuracy of %0.2f %%", methodName, valAcc * 100.0));

# Pegasos Method
methodName = "Pegasos";

mW = zeros(numFeat, numIterations + 1);

hXₖ!( vX :: Vector{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer} = copy!(vX, view(mK, :, ii));

mW = PegasosSVM!(mW, hXₖ!, vY, λ, numIterations);

dSolvers[methodName] = [hObjFun(mW[:, ii]) for ii ∈ 1:size(mW, 2)];

# Snacks Method
methodName = "Snacks";

mW = zeros(numFeat, numIterations + 1);

mW = SnacksSVM!(mW, hXₖ!, vY, λ, numRestarts, ω, η, ballRadius);

dSolvers[methodName] = [hObjFun(mW[:, ii]) for ii ∈ 1:size(mW, 2)];

# Solution 

vW = copy(vWRef);
vW = mW[:, end];
vŶ = [PredictSVM(vW, mK[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);

vXX = collect(LinRange(-2, 2, numGridPts));
mZ  = zeros(numGridPts, numGridPts);
vXₜ = zeros(dataDim);

for (jj, x1) in enumerate(vXX)
    for (ii, x2) in enumerate(vXX)
        vXₜ[1] = x1;
        vXₜ[2] = x2;
        vKx = EmbedData(reshape(vXₜ, (:, 1)), mC, sKc, σ);
        mZ[ii, jj] = PredictSVM(vW, vKx[:]);
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

titleStr =  @sprintf("Kernel SVM Classifier, Accuracy: %0.2f%%", 100.0 * valAcc);

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

