# StackOverflow Q60113001
# https://stackoverflow.com/questions/60113001
# Implementation of Edge Enhancing Diffusion (EED) with Structure Tensor.
# References:
#   1.  Dirk Jan Kroon - Image Edge Enhancing Coherence Filter Toolbox - https://www.mathworks.com/matlabcentral/fileexchange/25449.
#   2.  Nils Olovsson - Image Smearing by Anisotropic Diffusion (https://nils-olovsson.se/articles/image_smearing_by_anisotropic_diffusion).
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  Employing the Additive Operator Splitting (AOS) approach.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     03/09/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using ColorTypes;          #<! Required for Image Processing
using PlotlyJS;
using FileIO;              #<! Required for loading images
using LoopVectorization;   #<! Required for Image Processing
using StableRNGs;
using StaticKernels;       #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

@enum OpMode begin
    OP_MODE_EDGE_ENHANCING
    OP_MODE_COHERENCE_ENHANCING
    OP_MODE_DIRECTED_SMEARING_ALONG
    OP_MODE_DIRECTED_SMEARING_ACROSS
end

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function CalcImageGrad( mI :: Matrix{T}, mKx :: Matrix{T}, mKy :: Matrix{T}, σ :: T ) where {T <: AbstractFloat}

    STD_TO_RADIUS_FACTOR = T(4.0); #<! Match `scipy.ndimage.gaussian_filter` default
    
    mII = copy(mI);

    # Assuming `mKx`, `mKy` have the same dimensions
    padRadiusV = size(mKx, 1) ÷ 2; #<! Vertical
    padRadiusH = size(mKx, 2) ÷ 2; #<! Horizontal
    
    if (σ > zero(T))
        gaussKernelRadius = ceil(Int, STD_TO_RADIUS_FACTOR * σ);
        mG = GenGaussianKernel(σ, (gaussKernelRadius, gaussKernelRadius));
        mIIPad = PadArray(mII, (gaussKernelRadius, gaussKernelRadius), PAD_MODE_SYMMETRIC);
        Conv2D!(mII, mIIPad, mG; convMode = CONV_MODE_VALID);
    end

    mIIPad = PadArray(mII, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);

    mIx = Conv2D(mIIPad, mKx; convMode = CONV_MODE_VALID);
    mIy = Conv2D(mIIPad, mKy; convMode = CONV_MODE_VALID);

    return mIx, mIy;
    
end

function CalcImageStructureTensor( mIx :: Matrix{T}, mIy :: Matrix{T}, ρ :: T ) where {T <: AbstractFloat}

    STD_TO_RADIUS_FACTOR = T(4.0); #<! Match `scipy.ndimage.gaussian_filter` default
    
    numRows, numCols = size(mIx); #<! mIx and mIy have equal dimensions
    mJ = zeros(T, numRows, numCols, 3); #<! Symmetric

    mJ[:, :, 1] .= mIx .* mIx;
    mJ[:, :, 2] .= mIx .* mIy;
    mJ[:, :, 3] .= mIy .* mIy;

    if (ρ > zero(T))
        gaussKernelRadius = ceil(Int, STD_TO_RADIUS_FACTOR * ρ);
        mG = GenGaussianKernel(ρ, (gaussKernelRadius, gaussKernelRadius));
        for ii ∈ 1:size(mJ, 3)
            mJPad = PadArray(mJ[:, :, ii], (gaussKernelRadius, gaussKernelRadius), PAD_MODE_SYMMETRIC);
            Conv2D!(@view(mJ[:, :, ii]), mJPad, mG; convMode = CONV_MODE_VALID); #<! Requires a view
            # mJ[:, :, ii] = Conv2D(mJPad, mG; convMode = CONV_MODE_VALID);
        end
    end

    return mJ;
    
end

function CalcImageDiffusivityTensor( mJ :: Array{T, 3}, α :: T, opMode :: OpMode ) where {T <: AbstractFloat}

    C = T(3.315); #<! Constant by Joachim Weickert
    
    numRows = size(mJ, 1);
    numCols = size(mJ, 2);

    mJ11 = mJ[:, :, 1];
    mJ12 = mJ[:, :, 2];
    mJ21 = mJ[:, :, 2];
    mJ22 = mJ[:, :, 3];

    mJ11J22 = mJ11 - mJ22;
    mSqrt = sqrt.(mJ11J22 .* mJ11J22 .+ T(4.0) .* mJ12 .* mJ12);

    # Eigen Vector (A - 1st component, B - 2nd component)
    mV1A = T(2.0) * mJ12;
    mV1B = mJ22 .- mJ11 .+ mSqrt;

    # Normalize the directions
    mEigVecNorm = sqrt.(mV1A .* mV1A .+ mV1B .* mV1B);
    vI = mEigVecNorm .> T(1e-8); #<! TODO: Make a parameter / constant
    mV1A[vI] ./=  mEigVecNorm[vI];
    mV1B[vI] ./=  mEigVecNorm[vI];

    # Eigen Vector (A - 1st component, B - 2nd component)
    mV2A = copy(mV1B);
    mV2B = copy(-mV1A);

    # Eigen Values
    mμ₁ = T(0.5) .* (mJ11 + mJ22 + mSqrt);
    mμ₂ = T(0.5) .* (mJ11 + mJ22 - mSqrt);

    if (opMode == OP_MODE_EDGE_ENHANCING)
        mλ₁ = ones(T, numRows, numCols);
        mλ₂ = ones(T, numRows, numCols);
        vInd = abs.(mμ₁) .> T(1e-10); #<! Make a parameter / constant
        mλ₁Tmp = one(T) .- exp.(-C ./ ((mμ₁ .^ T(4.0)) .+ T(1e-10)));
        mλ₁[vInd] = mλ₁Tmp[vInd]
    elseif (opMode == OP_MODE_COHERENCE_ENHANCING)
        mλ₁ = ones(T, numRows, numCols);
        mλ₂ = ones(T, numRows, numCols);
        mDiff = abs.(mμ₁ .- mμ₂);
        vInd = mDiff .> T(1e-10); #<! Make a parameter / constant
        mλ₂Tmp = α .+ (one(T) .- α) .* exp.(-one(T) ./ (mDiff .* mDiff));
        mλ₂[vInd] = mλ₂Tmp[vInd];
    elseif (opMode == OP_MODE_DIRECTED_SMEARING_ALONG)
        mλ₁ = zeros(T, numRows, numCols);
        mλ₂ = T(0.9) * ones(T, numRows, numCols);
    elseif (opMode == OP_MODE_DIRECTED_SMEARING_ACROSS)
        mλ₁ = T(0.9) * ones(T, numRows, numCols);
        mλ₂ = zeros(T, numRows, numCols);
    end

    mD = zeros(T, numRows, numCols, 3); #<! Symmetric
    mD[:, :, 1] .= mλ₁ .* mV1A .* mV1A .+ mλ₂ .* mV2A .* mV2A;
    mD[:, :, 2] .= mλ₁ .* mV1A .* mV1B .+ mλ₂ .* mV2A .* mV2B;
    mD[:, :, 3] .= mλ₁ .* mV1B .* mV1B .+ mλ₂ .* mV2B .* mV2B;
    
    return mD;
    
end


function CalcTimeUpdate( mU :: Matrix{T}, mD :: Array{T, 3}, mKx :: Matrix{T}, mKxx :: Matrix{T}, mKy :: Matrix{T}, mKyy :: Matrix{T}, mKxy :: Matrix{T} ) where {T <: AbstractFloat}

    mA = mD[:, :, 1];
    mB = mD[:, :, 2];
    mC = mD[:, :, 2];
    mD = mD[:, :, 3];

    # Assumes `mKx`, `mKy` have the same support
    padRadiusV = size(mKx, 1) ÷ 2; #<! Vertical
    padRadiusH = size(mKx, 2) ÷ 2; #<! Horizontal

    mPad = PadArray(mA, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mAx = Conv2D(mPad, mKx; convMode = CONV_MODE_VALID);
    PadArray!(mPad, mB, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mBy = Conv2D(mPad, mKy; convMode = CONV_MODE_VALID);
    PadArray!(mPad, mC, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mCx = Conv2D(mPad, mKx; convMode = CONV_MODE_VALID);
    PadArray!(mPad, mD, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mDy = Conv2D(mPad, mKy; convMode = CONV_MODE_VALID);

    # TODO: Could be reused from previous calculation
    PadArray!(mPad, mU, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mUx = Conv2D(mPad, mKx; convMode = CONV_MODE_VALID);
    mUy = Conv2D(mPad, mKy; convMode = CONV_MODE_VALID);

    # Assumes `mKxx`, `mKyy` have the same support
    padRadiusV = size(mKxx, 1) ÷ 2; #<! Vertical
    padRadiusH = size(mKxx, 2) ÷ 2; #<! Horizontal

    mPad = PadArray(mU, (padRadiusV, padRadiusH), PAD_MODE_SYMMETRIC);
    mUxx = Conv2D(mPad, mKxx; convMode = CONV_MODE_VALID);
    mUyy = Conv2D(mPad, mKyy; convMode = CONV_MODE_VALID);
    mUxy = Conv2D(mPad, mKxy; convMode = CONV_MODE_VALID);

    mDiv∇U = ((mAx .+ mBy) .* mUx) .+ ((mCx .+ mDy) .* mUy);
    mTr∇U  = (mA .* mUxx) .+ (mD .* mUyy) .+ ((mB .+ mC) .* mUxy);

    return mDiv∇U + mTr∇U; #<! TODO: Optimize the allocation

end


## Parameters

# Data
imgUrl = raw"https://i.imgur.com/Y6Vr4OW.png"; #<! Alternative https://i.postimg.cc/j5VbqJkH/image.png

# Model

σ = 5.5; #<! Gaussian Kernel Standard Deviation for Image Smoothing
ρ = 5.5; #<! Gaussian Kernel Standard Deviation for Structure Tensor Smoothing
α = 0.1; #<! Diffusivity parameter (Effective only in `OP_MODE_COHERENCE_ENHANCING`)

opMode = OP_MODE_DIRECTED_SMEARING_ALONG;

# Kernels
# Look at A Scheme for "Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance" for deeper analysis of kernels
# Sobel Kernels
mKx     = -(1.0 /  8.0) .* [-1.0 0.0 1.0; -2.0 0.0 2.0; -1.0 0.0 1.0];
mKxx    =  (1.0 / 64.0) .* [ 1.0  0.0 -2.0 0.0 1.0; 4.0 0.0 -8.0 0.0 4.0; 6.0 0.0 -12.0 0.0 6.0; 4.0 0.0 -8.0 0.0 4.0; 1.0 0.0 -2.0 0.0 1.0];
mKxy    = -(1.0 / 64.0) .* [-1.0 -2.0 0.0 2.0 1.0; -2.0 -4.0 0.0 4.0 2.0; 0.0 0.0 0.0 0.0 0.0; 2.0 4.0 0.0 -4.0 -2.0; 1.0 2.0 0.0 -2.0 -1.0];
mKy     = collect(mKx'); #<! Explicit array
mKyy    = collect(mKxx'); #<! Explicit array
# Scharr Kernels
mKx     = -(1.0 /   32.0) .* [-3.0 0.0  3.0; -10.0 0.0 10.0; -3.0 0.0 3.0];
mKxx    =  (1.0 / 1024.0) .* [9.0 0.0 -18.0 0.0 9.0; 60.0 0.0 -120.0 0.0 60.0; 118.0 0.0 -236.0 0.0 118.0; 60.0 0.0 -120.0 0.0 60.0; 9.0 0.0 -18.0 0.0 9.0];
mKxy    = -(1.0 / 1024.0) .* [-9.0 -30.0 0.0 30.0 9.0; -30.0 -100.0 0.0 100.0 30.0; 0.0 0.0 0.0 0.0 0.0; 30.0 100.0 0.0 -100.0 -30.0; 9.0 30.0 0.0 -30.0 -9.0];
mKy     = collect(mKx'); #<! Explicit array
mKyy    = collect(mKxx'); #<! Explicit array

# Solver Parameters
τ = 0.1; #<! Time Step Size (Must be ≤ 0.125)
numIter = 100; #<! Iterations


#%% Load / Generate Data

mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = mI ./ 255.0;
if (ndims(mI) > 2) #<! Assumes RGB (Not RGBA)
    mI = mean(mI, dims = 3);
    mI = dropdims(mI, dims = 3);
end

numRows = size(mI, 1);
numCols = size(mI, 2);

numPx = numRows * numCols;


## Analysis

mU = copy(mI);

for ii ∈ 1:numIter
    # Explicit method loop
    mUx, mUy = CalcImageGrad(mU, mKx, mKy, σ);
    mJ = CalcImageStructureTensor(mUx, mUy, ρ);
    mD  = CalcImageDiffusivityTensor(mJ, α, opMode);
    m∂Uₜ = CalcTimeUpdate(mU, mD, mKx, mKxx, mKy, mKyy, mKxy);
    mU .+= τ .* m∂Uₜ;
end


## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    # savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mU; titleStr = "Output Image, σ = $(σ), ρ = $(ρ), α = $(α)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    # savefig(hP, figFileNme);
end