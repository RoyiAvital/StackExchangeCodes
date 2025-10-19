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

function PegasosSVM!( vW :: Vector{T}, hXᵢ! :: Function, vY :: Vector{T}, λ :: T, numIter :: N ) where {T <: AbstractFloat, N <: Integer}
    # Following Shai Shalev Shwartz, Shai Ben David - Understanding Machine Learning: From Theory to Algorithms.
    # See page 213 - SGD for Solving Soft SVM.
    # `hXₖ!(vZ, ii)` - Returns the `ii` -th data sample in `vZ`.

    numSamples  = length(vY);
    numFeatures = length(vW);

    vWₜ = zeros(T, numFeatures);
    vθₜ = zeros(T, numFeatures);
    vXₜ = zeros(T, numFeatures);

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
    end

    vW ./= T(numIter);

    return vW;

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

function SnacksSVM!( vW :: Vector{T}, hXᵢ! :: Function, vY :: Vector{T}, λ :: T, numRestarts :: N, numIter :: N, ω :: T, η :: T, ballRadius :: T ) where {T <: AbstractFloat, N <: Integer}
    # Snacks: A Fast Large Scale Kernel SVM Solver (https://arxiv.org/abs/2304.07983)

    numSamples  = length(vY);
    numFeatures = length(vW);

    vC  = zeros(T, numFeatures);
    vW̄  = zeros(T, numFeatures);
    vKᵢ = zeros(T, numFeatures);
    vG  = zeros(T, numFeatures);
    vB  = zeros(T, numFeatures);

    for kk in 1:numRestarts
        copy!(vC, vW);
        # copy!(vW̄, vW); #<! The paper
        vW̄ .= zero(T); #<! The code (See https://github.com/SofianeTanji/snacks/issues/2)

        for tt in 1:numIter
            ii = rand(1:numSamples);
            yᵢ = vY[ii];
            vKᵢ = hXᵢ!(vKᵢ, ii);
            valFlag  = (yᵢ * dot(vW, vKᵢ)) < one(T);
            @. vG = λ * vW - valFlag * yᵢ * vKᵢ;
            vW .-= η * vG;
            vB = ProjectL2Ball!(vB, vW, vC, ballRadius);
            vW̄ .+= vB ./ T(numIter);
        end
        copy!(vW, vW̄);
        ballRadius /= ω;
        η /= ω;
    end

    return vW;

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
csvFileName = raw"DataSet.csv"; #<! Binary Data

# Embedding
numCenters = 250;

# SVM Model
σ = 0.025;
λ = 0.1;
α = 1e-5; #<! Regularization to make `mK` SPD

# Solvers
# Pegasos
numIterations = 1_000;

# Snacks
numRestarts = 10; #<! K
numInIter   = 100; #<! T
ballRadius  = 4.0;
ω           = 2.0; #<! Shrinking factor
η           = 1.0; #<! Initial step size

## Load / Generate Data

mD = readdlm(csvFileName, ',', Float64; skipstart = 1);
mX = collect(mD[:, 1:2]'); #<! Sample in a row -> Sample in a column
vY = mD[:, end];

# Shift to Center
mX .-= mean(mX; dims = 2);

dataDim    = size(mX, 1);
numSamples = size(mX, 2);

vCenIdx = sample(oRng, 1:numSamples, numCenters; replace = false);
mC      = mX[:, vCenIdx];
numFeat = numCenters;


## Analysis

# Create approximated embeddings
# Once the embeddings are available, the problem can be solved using LinearSVM
mKc, sKc = ApproxEmbedder(mC, σ; α = α);
mK = EmbedData(mX, mC, sKc, σ);
hObjFun(vW :: Vector{T}) where {T <: AbstractFloat} = SVM(vW, mK, vY, λ);


# DCP Solver
methodName = "Convex.jl"

vWRef  = SVMCVX(mK, vY, λ);
optVal = hObjFun(vWRef);

vŶ = [PredictSVM(vWRef, mK[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);
println(@sprintf("%s: Accuracy of %0.2f %%", methodName, valAcc * 100.0));

# Pegasos Method
methodName = "Pegasos";

vW = zeros(numFeat);

hXₖ!( vX :: Vector{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer} = copy!(vX, view(mK, :, ii));

vW = PegasosSVM!(vW, hXₖ!, vY, λ, numIterations);

vWPegasos = copy(vW);

vŶ = [PredictSVM(vWPegasos, mK[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);
println(@sprintf("%s: Accuracy of %0.2f %%", methodName, valAcc * 100.0));
runTime = @belapsed PegasosSVM!($vW, $hXₖ!, $vY, $λ, $numIterations);
println(@sprintf("%s: Run Time of %0.2f [Mili Sec]", methodName, runTime * 1000.0));

# Snacks Method
methodName = "Snacks";

vW = zeros(numFeat);

vW = SnacksSVM!(vW, hXₖ!, vY, λ, numRestarts, numInIter, ω, η, ballRadius);

vWSnacks = copy(vW);

vŶ = [PredictSVM(vWSnacks, mK[:, ii]) for ii in 1:numSamples];
valAcc = mean(vY .== vŶ);
println(@sprintf("%s: Accuracy of %0.2f %%", methodName, valAcc * 100.0));
runTime = @belapsed SnacksSVM!($vW, $hXₖ!, $vY, $λ, $numRestarts, $numInIter, $ω, $η, $ballRadius);
println(@sprintf("%s: Run Time of %0.2f [Mili Sec]", methodName, runTime * 1000.0));

