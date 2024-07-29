# StackExchange Computational Science Q24016744
# https://stackoverflow.com/questions/24016744
# Creating Filter's Laplacian Matrix and Solving the Linear Equation for Image Filtering.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  Using `Symmetric(BandedMatrix(mA));` for teh 5 point operator seems to be slow in Julia.  
#       Namely, it has no performance improvement over using the direct solver on `mA` itself.
#       It takes ~500 [Sec].
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     20/07/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BandedMatrices;
using BenchmarkTools;
using ColorTypes;          #<! Required for Image Processing
using ConjugateGradients;
using FileIO;              #<! Required for loading images
using Krylov;
using LimitedLDLFactorizations;
using LinearOperators;
using LoopVectorization;   #<! Required for Image Processing
using MKLSparse;
using PlotlyJS;
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

struct DerivativeKernels{T <: AbstractFloat}
    mKx  :: Matrix{T}
    mK⁺x :: Matrix{T}
    mKy  :: Matrix{T}
    mK⁺y :: Matrix{T}
    DerivativeKernels{T}() where {T <: AbstractFloat} = new([one(T) -one(T)], [-one(T) one(T)], [one(T); -one(T);;], [-one(T); one(T);;]);
end

DerivativeKernels(dataType :: Type{<: AbstractFloat} = Float64) = DerivativeKernels{dataType}();

## Parameters

imgUrl = raw"https://i.imgur.com/69CLVcn.png" #<! "Flower.png" locally

λ = 0.05;
# Using `γ` instead of `γ` as `γ` reserved for the BLAS operation
γ = 0.95;
ε = 1e-5;

# Solver Parameters


#%% Load / Generate Data

mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = mI ./ 255.0;

numRows = size(mI, 1);
numCols = size(mI, 2);

numPx = numRows * numCols;

## Analysis
# The matrix: (C + D + λ * I) * x = y => A x = y

# Define the A operator

function OpAMul!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T, sK :: DerivativeKernels{T}, λ :: T, γ :: T, ε :: T, mO :: AbstractMatrix{T} ) where {T <: AbstractFloat}

    # Using `λ` instead of `α` as `α` reserved for the BLAS operation
    γ² = γ * γ;
    
    mX = @invoke reshape(vX, size(mO));

    mDx = Conv2D(mX, sK.mKx; convMode = CONV_MODE_VALID);
    # mAx = exp.(-(mDx .* mDx) ./ (2γ²));
    mAx = inv.((abs.(mDx) .^ γ) .+ ε)
    mDx .*= mAx;
    mDx = Conv2D(mDx, sK.mK⁺x; convMode = CONV_MODE_FULL);
    
    mDy = Conv2D(mX, sK.mKy; convMode = CONV_MODE_VALID);
    # mAy = exp.(-(mDy .* mDy) ./ (2γ²));
    mAy = inv.((abs.(mDy) .^ γ) .+ ε)
    mDy .*= mAy;
    mDy = Conv2D(mDy, sK.mK⁺y; convMode = CONV_MODE_FULL);

    mO .= mX .+ λ .* (mDx .+ mDy);
    if (β == zero(T))
        if (α == zero(T))
            vY .= zero(T);
        elseif (α == one(T))
            copyto!(vY, mO); #<! `copyto!()` ignores the shape
        else
            vY .= α .* mO[:];
        end
    elseif (β == one(T))
        if (α == one(T))
            vY .+= mO[:];
        else
            vY .+= α .* mO[:];
        end
    else
        if (α == zero(T))
            vY .*= β;
        elseif (α == one(T))
            vY .= (β .* vY) .+ mO[:];
        else
            vY .= (β .* vY) .+ α .* mO[:];
        end
    end

end

function OpAMul!!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T, sK :: DerivativeKernels{T}, λ :: T, γ :: T, ε :: T, mTx :: AbstractMatrix{T}, mTy :: AbstractMatrix{T}, mDx :: AbstractMatrix{T}, mDy :: AbstractMatrix{T}, mO :: AbstractMatrix{T} ) where {T <: AbstractFloat}

    # Using `λ` instead of `α` as `α` reserved for the BLAS operation
    
    mX = @invoke reshape(vX, size(mO));

    mTx = Conv2D!(mTx, mX, sK.mKx; convMode = CONV_MODE_VALID);
    mTx ./= (abs.(mTx) .^ γ) .+ ε;
    mDx = Conv2D!(mDx, mTx, sK.mK⁺x; convMode = CONV_MODE_FULL);
    
    mTy = Conv2D!(mTy, mX, sK.mKy; convMode = CONV_MODE_VALID);
    mTy ./= (abs.(mTy) .^ γ) .+ ε;
    mDy = Conv2D!(mDy, mTy, sK.mK⁺y; convMode = CONV_MODE_FULL);

    mO .= mX .+ λ .* (mDx .+ mDy);
    if (β == zero(T))
        if (α == zero(T))
            vY .= zero(T);
        elseif (α == one(T))
            copyto!(vY, mO); #<! `copyto!()` ignores the shape
        else
            vY .= α .* mO[:];
        end
    elseif (β == one(T))
        if (α == one(T))
            vY .+= mO[:];
        else
            vY .+= α .* mO[:];
        end
    else
        if (α == zero(T))
            vY .*= β;
        elseif (α == one(T))
            vY .= (β .* vY) .+ mO[:];
        else
            vY .= (β .* vY) .+ α .* mO[:];
        end
    end

end

function GenMatA( vY :: AbstractVector{T}, numRows :: N, numCols :: N, λ :: T, γ :: T, ε :: T ) where {T <: AbstractFloat, N <: Integer}

    mDx = GenConvMtx([1.0 -1.0], numRows, numCols; convMode = CONV_MODE_VALID);
    mDy = GenConvMtx([1.0; -1.0;;], numRows, numCols; convMode = CONV_MODE_VALID);
    
    # mAx = Diagonal(exp.(- ((mDx * vY) .^ 2) ./ (2 * γ * γ)));
    # mAy = Diagonal(exp.(- ((mDy * vY) .^ 2) ./ (2 * γ * γ)));
    
    # Could apply λ in this step
    mAx = Diagonal(inv.((abs.(mDx * vY) .^ γ) .+ ε));
    mAy = Diagonal(inv.((abs.(mDy * vY) .^ γ) .+ ε));
    
    mA = (I + λ * (mDy' * mAy * mDy + mDx' * mAx * mDx));

    return mA;

end

function GenMatAxAy( vY :: AbstractVector{T}, numRows :: N, numCols :: N, λ :: T, γ :: T, ε :: T ) where {T <: AbstractFloat, N <: Integer}

    mDx = GenConvMtx([1.0 -1.0], numRows, numCols; convMode = CONV_MODE_VALID);
    mDy = GenConvMtx([1.0; -1.0;;], numRows, numCols; convMode = CONV_MODE_VALID);
    
    # mAx = Diagonal(exp.(- ((mDx * vY) .^ 2) ./ (2 * γ * γ)));
    # mAy = Diagonal(exp.(- ((mDy * vY) .^ 2) ./ (2 * γ * γ)));
    
    # Could apply λ in this step
    mAx = Diagonal(inv.((abs.(mDx * vY) .^ γ) .+ ε));
    mAy = Diagonal(inv.((abs.(mDy * vY) .^ γ) .+ ε));

    return mAx, mAy;

end

sK = DerivativeKernels(Float64);

mO = similar(mI);
mTx = zeros(numRows, numCols - 1);
mTy = zeros(numRows - 1, numCols);
mDx = similar(mI);
mDy = similar(mI);
vY = zeros(numPx);
copyto!(vY, mI); #<! `copyto!()` ignores the shape

# No need for `MulAT!()` as the operator is symmetric
MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, sK, λ, γ, ε, mO);
MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!!(vY, vX, α, β, sK, λ, γ, ε, mTx, mTy, mDx, mDy, mO);

hOpA( vY :: AbstractVector{T}, vX :: AbstractVector{T} ) where {T <: AbstractFloat} = OpAMul!!(vY, vX, 1.0, 0.0, sK, λ, γ, ε, mTx, mTy, mDx, mDy, mO);
oCG = CGData(numPx, eltype(vY));

vX = copy(vY);
# vX, exitCode, numItr = ConjugateGradients.cg(hOpA, vY; tol = 5e-7, maxIter = 3_000, data = oCG);
exitCode, numItr = ConjugateGradients.cg!(hOpA, vY, vX; tol = 5e-7, maxIter = 3_000, data = oCG);
# exitCode, numItr = ConjugateGradients.cg!(hOpA, vY, vX; tol = 5e-7, maxIter = 3_000, precon = (x, y) -> ldiv!(x, mP, y), data = oCG);
runTime = @belapsed ConjugateGradients.cg!(hOpA, vY, vX; tol = 5e-7, maxIter = 3_000, data = oCG) setup = (vX = copy(vY)) seconds = 2;
resAnalysis = @sprintf("Conjugate Gradient solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

# The operator is SPD (Symmetric Positive Definite)
isSymmetric = true;
isHermitian = false; #<! Irrelevant
opA = LinearOperator(Float64, numRows * numCols, numRows * numCols, isSymmetric, isHermitian, MulA!);

#  Preconditioner
mA = GenMatA(vY, numRows, numCols, λ, γ, ε);
mP = lldl(mA);
mP.D .= abs.(mP.D);

mD = Diagonal(mA);
mP = Diagonal(inv.(mD.diag));

oKrylovSolver = CgSolver(opA, vY);
vX = copy(vY);
Krylov.cg!(oKrylovSolver, opA, vY, vX);
runTime = @belapsed Krylov.cg!(oKrylovSolver, opA, vB, vX) setup = (vB = copy(vY)) seconds = 2;
resAnalysis = @sprintf("Conjugate Gradient solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

vX = oKrylovSolver.x;
copyto!(mO, vX);
clamp!(mO, 0.0, 1.0);

mA = GenMatA(vY, numRows, numCols, λ, γ, ε);
vXD = mA \ vY;
runTime = @belapsed GenMatA(vY, numRows, numCols, λ, γ, ε) \ vB setup = (vB = copy(vY)) seconds = 2;
resAnalysis = @sprintf("Direct solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

copyto!(mO, vXD);
clamp!(mO, 0.0, 1.0);

# Banded Matrix
# Seems to be much slower!
# mA = GenMatA(vY, numRows, numCols, λ, γ, ε);
# mB = Symmetric(BandedMatrix(mA));
# vXD = mB \ vY;
# runTime = @belapsed Symmetric(BandedMatrix(GenMatA(vY, numRows, numCols, λ, γ, ε))) \ vB setup = (vB = copy(vY)) seconds = 2;
# resAnalysis = @sprintf("Direct solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

# copyto!(mO, vXD);
# clamp!(mO, 0.0, 1.0);


# Timing Operators
runTime = @belapsed GenMatA(vY, numRows, numCols, λ, γ, ε) * vB setup = (vB = copy(vY)) seconds = 2;
resAnalysis = @sprintf("Matrix operator run time: %0.5f [Sec]", runTime);
println(resAnalysis);


runTime = @belapsed opA * vB setup = (vB = copy(vY)) seconds = 2;
resAnalysis = @sprintf("Functional operator run time: %0.5f [Sec]", runTime);
println(resAnalysis);





## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image, λ = $(λ)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# for (ii, λ) in enumerate(vλ)
#     MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, λ, mO, mB, mV1, mV2, mKₕ, mKᵥ);
#     local opA = LinearOperator(Float64, numRows * numCols, numRows * numCols, true, false, MulA!);
#     local vB = 1.0 .+ λ .* (mDₕ' * mDₕ + mDᵥ' * mDᵥ) * vY;
#     cg!(oCgSolve, opA, vB);
#     copyto!(mX, oCgSolve.x);
#     local hP = DisplayImage(mX; titleStr = "λ = $λ");
#     display(hP);

#     global figureIdx += 1;
    
#     if (exportFigures)
#         local figFileNme = @sprintf("Figure%04d.png", figureIdx);
#         savefig(hP, figFileNme);
#     end
# end

# Run Time Analysis
# runTime = @belapsed cgls!(oCglsSolver, opA, vB) setup = (vB = copy(vY)) seconds = 2;
# resAnalysis = @sprintf("The CG solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

# mA = (mC + mD + mI);
# runTime = @belapsed (mA \ vY) seconds = 2;
# resAnalysis = @sprintf("The Direct solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);


# using MAT;

# dVars = matread("Data.mat");
# mI = dVars["mI"];
# mK = dVars["mK"];
# mP = dVars["mP"];
# mY = dVars["mY"];

# numRows = size(mI, 1);
# numCols = size(mI, 2);

# numRowsK = size(mK, 1);
# numColsK = size(mK, 2);

# tuPadRadius = (numRowsK ÷ 2, numColsK ÷ 2);
# mPP = PadArray(mI, tuPadRadius; padMode = PAD_MODE_CIRCULAR);
# mYY = Conv2D(mPP, mK; convMode = CONV_MODE_VALID);

# mH = GenFilterMtx(mK, numRows, numCols; filterMode = FILTER_MODE_CONVOLUTION, boundaryMode = BND_MODE_CIRCULAR);

# norm(mP - mPP, Inf)
# norm(mY - mYY, Inf)

# norm(mY[:] - mH * mI[:], Inf)
