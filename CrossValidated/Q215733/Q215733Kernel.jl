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

function KernelSVM( vβ :: Vector{T}, mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}
    # The Kernel SVM without the bias term

    numSamples = length(vY);
    
    # Regularization
    objVal = 0.5 * λ * dot(vβ, mK, vβ);

    # Objective
    vH = [max(zero(T), T(1) - vY[ii] * dot(vβ, mK[:, ii])) for ii in 1:numSamples];
    if squareHinge
        vH = [vH[ii] * vH[ii] for ii in 1:numSamples];
    end

    objVal += sum(vH) / numSamples;

    # Vectorized form
    # objVal = 0.5 * λ * dot(vβ, mK, vβ) + sum(max.(zero(T), T(1) .- vY .* (mK * vβ)));

    return objVal;

end

function KernelSVMCVX( mK :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # Olivier Chapelle - Training a Support Vector Machine in the Primal
    # The Kernel SVM without the bias term

    # `mK` the kernel matrix
    numSamples = size(mK, 1);

    vβ = Convex.Variable(numSamples);

    hingeLoss = Convex.sum(Convex.pos(T(1) - vY .* (mK * vβ)));
    hingeLoss = hingeLoss / numSamples; #<! Stochastic sample only one of the corpus

    sConvProb = minimize( T(0.5) * λ * Convex.quadform(vβ, mK; assume_psd = true) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vβ.value);
    
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

function PegasosKernelSVM!( mX :: Matrix{T}, hKₖ :: Function, vY :: Vector{T}, λ :: T, vαₜ :: Vector{T}, vβₜ :: Vector{T} ) where {T <: AbstractFloat}
    # Following Shai Shalev Shwartz, Shai Ben David - Understanding Machine Learning: From Theory to Algorithms.
    # See page 223 - SGD for Solving Soft SVM with Kernels.
    # `hKₖ(vZ, ii)` - Returns the dot product of the `ii` -th data sample with `vZ`.

    numSamples    = length(vY);
    dataDim       = size(mX, 1);
    numIterations = size(mX, 2);

    # First Iteration
    ii = 1;
    tt = ii;
    ηₜ = inv(λ * tt);

    vαₜ .= ηₜ .* vβₜ;

    kk = rand(1:numSamples);
    yₖ = vY[kk];

    valSum  = yₖ * hKₖ(vαₜ, kk);
    vβₜ[kk] += (valSum < one(T)) * yₖ;

    @views mX[:, ii] = vαₜ;

    for ii in 2:numIterations

        tt = ii;
        ηₜ = inv(λ * tt);

        vαₜ .= ηₜ .* vβₜ;

        kk = rand(1:numSamples);
        yₖ = vY[kk];

        # @views valSum = yₖ * dot(mK[:, kk], vαₜ);
        valSum  = yₖ * hKₖ(vαₜ, kk);
        vβₜ[kk] += (valSum < one(T)) * yₖ;

        @views mX[:, ii] .= inv(tt) .* ((T(ii - 1) .* mX[:, ii - 1]) .+ vαₜ);
    end

    return mX;

end

function PredictKernelSVM( vβ :: Vector{T}, vX :: Vector{T}, mX :: Matrix{T}, vY :: Vector{T}, σ :: T ) where {T <: AbstractFloat}
    # Treat any point as "Out of Sample" data

    numSamples = length(vβ);

    γ = inv(T(2) * σ * σ);
    valSum = zero(T);

    for ii in 1:numSamples
        # If `vβ` was the dual it should have been: `valSum += vβ[ii] * vY[ii] * CalcRbfKernel(vX, view(mX, :, ii), γ);`
        valSum += vβ[ii] * CalcRbfKernel(vX, view(mX, :, ii), γ);
    end

    return sign(valSum);

end


## Parameters

# Data
csvFileName = raw"BinaryClassificationData.csv";

# SVM Model
σ = 0.025;
λ = 0.1;
α = 1e-5; #<! Regularization to make `mK` SPD

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
mX .-= mean(mX; dims = 2);

dataDim    = size(mX, 1);
numSamples = size(mX, 2);

mK = CalcRbfKernelMat(mX, σ; α = α);

hObjFun(vβ :: Vector{T}) where {T <: AbstractFloat} = KernelSVM(vβ, mK, vY, λ);

dSolvers = Dict();

## Analysis

# DCP Solver
methodName = "Convex.jl"

vβRef  = KernelSVMCVX(mK, vY, λ);
optVal = hObjFun(vβRef);

dSolvers[methodName] = optVal * ones(numIterations);


# Pegasos Method
methodName = "Pegasos";

vαₜ = zeros(numSamples);
vβₜ = zeros(numSamples);

hKᵢ( vX :: Vector{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer} = dot(vX, view(mK, :, ii));

mβ = zeros(numSamples, numIterations);

mβ = PegasosKernelSVM!(mβ, hKᵢ, vY, λ, vαₜ, vβₜ);

dSolvers[methodName] = [hObjFun(mβ[:, ii]) for ii ∈ 1:size(mβ, 2)];

vβ = mβ[:, end];
vŶ = [PredictKernelSVM(vβ, mX[:, ii], mX, vY, σ) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);

vXX = collect(LinRange(-2, 2, numGridPts));
mZ  = zeros(numGridPts, numGridPts);
vXₜ = zeros(dataDim);

for (jj, x1) in enumerate(vXX)
    for (ii, x2) in enumerate(vXX)
        vXₜ[1] = x1;
        vXₜ[2] = x2;
        mZ[ii, jj] = PredictKernelSVM(vβ, vXₜ, mX, vY, σ);
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

