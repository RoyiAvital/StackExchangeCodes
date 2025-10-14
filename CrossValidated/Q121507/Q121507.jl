# StackExchange Cross Validated Q121507
# https://stats.stackexchange.com/questions/121507
# Solve the Kernel SVM with Zero Bias / Intercept Term.
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
# - 1.0.000     14/10/2025  Royi Avital
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

    objVal += sum(vH);

    # Vectorized form
    # objVal = 0.5 * λ * dot(vβ, mK, vβ) + sum(max.(zero(T), T(1) .- vY .* (mK * vβ)));

    return objVal;

end

function SolveCVX( mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}
    # Olivier Chapelle - Training a Support Vector Machine in the Primal
    # The Kernel SVM without the bias term

    # `mK` the kernel matrix
    numSamples = size(mK, 1);

    vβ     = Convex.Variable(numSamples);

    # Loop Form
    # vH = [Convex.pos(T(1) - vY[ii] * (Convex.dot(vβ, mK[:, ii]))) for ii in 1:numSamples];
    # if squareHinge
    #     vH = [Convex.square(vH[ii]) for ii in 1:numSamples];
    # end
    # hingeLoss = Convex.sum(vcat(vH...));

    # Vectorized Form
    if squareHinge
        hingeLoss = T(0.5) * Convex.sum(Convex.square(Convex.pos(T(1) - vY .* (mK * vβ))));
    else
        hingeLoss = Convex.sum(Convex.pos(T(1) - vY .* (mK * vβ)));
    end

    hingeLoss = hingeLoss / numSamples; #<! Stochastic sample only one of the corpus

    sConvProb = minimize( T(0.5) * λ * Convex.quadform(vβ, mK; assume_psd = true) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vβ.value);
    
end

function CalcRbfKernel( mX :: Matrix{T}, σ :: T; α :: T = zero(T) ) where {T <: AbstractFloat}
    # Calculates the RBF Kernel Matrix
    
    dataDim = size(mX, 1);
    numSamples = size(mX, 2);
    mK = ones(T, numSamples, numSamples);
    vD = zeros(T, dataDim);

    γ = inv(T(2) * σ * σ);

    for jj in 2:numSamples, ii in 1:(jj - 1)
        vD         .= view(mX, :, ii) .- view(mX, :, jj);
        mK[ii, jj]  = exp(-γ * sum(abs2, vD));
        mK[jj, ii]  = mK[ii, jj];
    end

    mK .+= α .* I(numSamples);

    return mK;

end

function ProxHingeLoss( vY :: Vector{T}, vZ :: Vector{T}, γ :: T ) where {T <: AbstractFloat}
    # Calculates \arg \min_{\boldsymbol{x}} \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} + \gamma \max \left\{ 0, 1 - \boldsymbol{z}^{T} \boldsymbol{x} \right\}
    
    valV = one(T) - dot(vZ, vY);
    sumSqrZ = sum(abs2, vZ);

    if valV < zero(T)
        vX = copy(vY);
    elseif valV > γ * sumSqrZ
        vX = vY + γ * vZ;
    else
        vX = vY + (valV / sumSqrZ) * vZ;
    end

    return vX;

end

function ProxHingeLossCvx( vY :: Vector{T}, vZ :: Vector{T}, γ :: T ) where {T <: AbstractFloat}
    # Calculates \arg \min_{\boldsymbol{x}} \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} + \gamma \max \left\{ 0, 1 - \boldsymbol{z}^{T} \boldsymbol{x} \right\}
    
    numElements = length(vY);

    vX = Convex.Variable(numElements);

    sConvProb = minimize( T(0.5) * Convex.sumsquares(vX - vY) + γ * Convex.pos(1 - Convex.dot(vZ, vX)) ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);

end


## Parameters

# Data
csvFileName = raw"BinaryClassificationData.csv";

# SVM Model
σ = 1.0;
λ = 0.1;
α = 1e-5; #<! Regularization to make `mK` SPD

# Solvers
numIterations = 50_000;
η             = 1e-4;

## Load / Generate Data

mD = readdlm(csvFileName, ',', Float64; skipstart = 1);
mX = collect(mD[:, 1:2]'); #<! Sample in a row -> Sample in a column
vY = mD[:, 3];
vY .= 2.0 .* vY .- 1.0; #<! Map {0, 1} -> {-1, 1}

dataDim    = size(mX, 1);
numSamples = size(mX, 2);

mK = CalcRbfKernel(mX, σ; α = α);

hObjFun(vβ :: Vector{T}) where {T <: AbstractFloat} = KernelSVM(vβ, mK, vY, λ);

dSolvers = Dict();

## Analysis

# DCP Solver
methodName = "Convex.jl"

vβRef  = SolveCVX(mK, vY, λ);
optVal = hObjFun(vβRef);

dSolvers[methodName] = optVal * ones(numIterations);

# Verify the Prox

vYY = randn(numSamples);
vZZ = randn(numSamples);
γ  = 1.0;

vXXRef = ProxHingeLossCvx(vYY, vZZ, γ);
vXX    = ProxHingeLoss(vYY, vZZ, γ);

println(sum(abs2, vXX - vXXRef));


# Stochastic Proximal Method
methodName = "Stochastic PGM";

hGradF(vX :: Vector{T}) where {T <: AbstractFloat} = λ * mK * vX;
hProxG(vY :: Vector{T}, vZ :: Vector{T}, γ :: T) where {T <: AbstractFloat} = ProxHingeLoss(vY, vZ, γ);

mX  = zeros(numSamples, numIterations);
vX  = zeros(numSamples);
vZ  = zeros(numSamples);
vT  = zeros(numSamples);

# Stochastic PGM
for ii in 2:numIterations
    global vX, η;
    copy!(vX, view(mX, :, ii - 1));
    kk = rand(oRng, 1:numSamples);
    yₖ = vY[kk];
    vKₖ = view(mK, :, kk);
    vZ .= yₖ .* vKₖ;

    vX = hProxG(vX .- η * hGradF(vX), vZ, 1.0 * η);
    mX[:, ii] .= vX;
end

# Accelerated (Does not work)
# for ii in 2:numIterations
#     global vX, vT, η;

#     kk = rand(oRng, 1:numSamples);
#     yₖ = vY[kk];
#     vKₖ = view(mK, :, kk);
#     vZ .= yₖ .* vKₖ;

#     vX .= hProxG(vT .- η * hGradF(vT), vZ, 1.0 * η);
#     fistaStepSize = (ii - 1) / (ii + 2);
#     # fistaStepSize = 0.0;
#     vT .= vX .+ fistaStepSize .* (vX .- view(mX, :, ii - 1));

#     # η = η * 0.97;

#     mX[:, ii] .= vX;
# end

# Pegasos (Fastest)
# mX  = zeros(numSamples, numIterations);
# vβₜ = zeros(numSamples);
# vαₜ = zeros(numSamples);
# for ii in 2:numIterations
#     global vαₜ, vβₜ;

#     tt = ii - 1;
#     η = inv(λ * tt);

#     vαₜ .= η .* vβₜ;

#     kk = rand(oRng, 1:numSamples);
#     yₖ = vY[kk];

#     @views valSum = yₖ * dot(mK[:, kk], vαₜ);
#     vβₜ[kk] += (valSum < 1.0) * yₖ;

#     @views mX[:, ii] .= (0.5 .* mX[:, ii - 1]) + (0.5 .* vαₜ);
# end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

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
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end