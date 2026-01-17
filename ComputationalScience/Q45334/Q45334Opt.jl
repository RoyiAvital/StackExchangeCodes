# StackExchange Computational Science Q45334
# https://scicomp.stackexchange.com/questions/45334
# Solving Regularized Least Squares with Linear Equality and Linear Inequality Constraints for Curve Smoothing.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     17/01/2026  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using Convex;
using ECOS;
# using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaArrays.jl")); #<! Display Images
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(RNG_SEED);
seedNum = Int(mod(round(time()), 5000));
oRng = StableRNG(seedNum);
oRng = StableRNG(1220);

## Types & Structs

const SpMat{T} = SparseMatrixCSC{T};

## Functions

function CVXSolver( mP :: MAT, vQ :: Vector{T}, mA :: MAT, vB :: Vector{T}, mC :: MAT, vD :: Vector{T} ) where {T <: AbstractFloat, MAT <: SpMat{T}}
    # `mP` assumed to be symmetric positive definite (SPD)

    dataDim = size(mP, 1);

    sC = cholesky(mP; check = false, perm = 1:dataDim);
    mU = sparse(sparse(sC.L)');

    vX = Convex.Variable(dataDim);
    # sConvProb = minimize( T(0.5) * Convex.quadform(vX, mP; assume_psd = true) + Convex.dot(vQ, vX), [mA * vX == vB, mC * vX <= vD] ); #<! See https://github.com/jump-dev/Convex.jl/issues/725
    sConvProb = minimize( T(0.5) * Convex.square(Convex.norm2(mU * vX)) + Convex.dot(vQ, vX), [mA * vX == vB, mC * vX <= vD] ); #<! See https://github.com/jump-dev/Convex.jl/issues/725
    solve!(sConvProb, ECOS.Optimizer; silent = true); #<! Faster for this problem
    # solve!(sConvProb, SCS.Optimizer; silent = true);
    
    return vec(vX.value);
    
end

function SplineQPSmoothCVX( vY :: Vector{T}, mD :: SpMat{T}, λ :: T, vI :: Vector{N}, mA :: SpMat{T} ) where {T <: AbstractFloat, N <: Integer}

    mP = I + λ * (mD' * mD);
    vQ = -vY;
    
    mA = sparse(1:numRefPts, vI, 1.0, numRefPts, numSamples); #<! Equality Constraints
    vB = vY[vI];
    mC = GenMonoOp(vI, vY); #<! Inequality Constraints
    # mC.nzval .= 0.0; #<! Disable Monotonicity Constraints
    vD = zeros(size(mC, 1));

    vX = CVXSolver(mP, vQ, mA, vB, mC, vD);

    return vX;

end

function SplineQPSmooth( vY :: Vector{T}, mD :: SpMat{T}, λ :: T, vI :: Vector{N}, mA :: SpMat{T}; ρ :: T = 1.0, numItr :: N = N(5_000), ϵAbs :: T = T(1e-5), ϵRel :: T = T(1e-5), convInterval :: N = N(25), τ :: T = T(10.0) ) where {T <: AbstractFloat, N <: Integer}
    # Solves:
    # \arg \min_x 0.5 || y - x ||_2^2 + λ || D x ||_2^2
    # s.t. x_i = y_i, ∀ i ∈ vI
    #      A x <= 0
    # Solve the problem using ADMM

    # This variation breaks the problem into:
    # vJ = {1, 2, 3, ...} \ vI
    # Then: D = [D_J, D_I], A = [A_J, A_I]
    # Then one can only optimize for the indices in `vJ`


    numSamples = length(vY);
    numEq      = length(vI);
    numInEq    = size(mA, 1);

    # Extract `vJ` as the set subtraction of `1:numSamples` and `vI`
    vJ = setdiff(1:numSamples, vI);

    mAi = mA[:, vI];
    mAj = mA[:, vJ];
    mDi = mD[:, vI];
    mDj = mD[:, vJ];

    # Quadratic Terms (Equality is implicit)
    mP = I + λ * (mDj' * mDj);
    vQ = -vY[vJ] + λ * (mDj' * (mDi * vY[vI]));
    vB = -mAi * vY[vI];

    ρ¹ = inv(ρ);

    # ADMM Variables
    vX  = copy(vY);       #<! Optimization variable
    vXj = vX[vJ];     #<! Optimization variable for `vJ` only
    vR  = zeros(length(vJ));       #<! Right Hand Side
    vS  = zeros(numInEq); #<! Slack variable for inequality
    vS1 = zeros(numInEq); #<! Previous iteration buffer
    vμ  = zeros(numInEq); #<! Dual variable for inequality (`vMu`)
    vZ  = copy(vμ);       #<! Auxiliary variable

    # Factorize the KKT System
    # (P + rho * A' * A) * z = r
    mAjAj = mAj' * mAj; 
    mK  = mP + ρ * mAjAj;
    mP  = AlignSparsePattern(mK, mP);
    mAjAj = AlignSparsePattern(mK, mAjAj);
    sK  = cholesky(Symmetric(mK); check = false);

    isConv   = false;
    updatedΡ = false;

    for ii = 1:numItr
        copy!(vS1, vS); #<! Previous iteration
        
        # Solve the Linear System
        # vR = -vQ + ρ * mAj' * (vB - vS - vμ); #<! Right hand vector
        @. vR = -vQ;
        @. vZ = vB - vS - vμ;
        mul!(vR, mAj', vZ, ρ, one(T));

        # vX = sK \ vR;
        ldiv!(vXj, sK, vR);
        
        # Proximal / Projection Step
        # s = vB -A * z with s >= 0
        # vS = max.(zero(T), -(mA * vX + ρ¹ * vμ));
        @. vS = vB - vμ;
        mul!(vS, mAj, vXj, -one(T), one(T));
        @. vS = max(vS, zero(T));
        
        # Update Dual Variables
        # vμ .+= (mAj * vXj + vS - vB);
        @. vμ += vS - vB;
        mul!(vμ, mAj, vXj, one(T), one(T));
        
        # Check Convergence
        if mod(ii, convInterval) == 0
            # primRes = norm(mA * vX + vS - vB, Inf);
            @. vZ = vS - vB;
            mul!(vZ, mAj, vXj, one(T), one(T));
            primRes = norm(vZ, Inf);
            # dualRes = norm(ρ * mA' * (vS - vS1), Inf);
            @. vZ = ρ * (vS - vS1);
            mul!(vR, mAj', vZ, one(T), zero(T));
            dualRes = norm(vR, Inf);
            
            if ((primRes < ϵAbs) && (dualRes < ϵAbs))
                isConv = true;
                break;
            end
            
            # Adapt `ρ`
            resRatio = primRes / dualRes;
            # fprintf('Primal Residual: %0.7f, Dual Residual: %0.7f\n', primRes, dualRes);
            # fprintf('Residual Ratio: %0.2f, ρ = %0.3f\n', resRatio, paramRho);
            if (resRatio > τ) || (inv(resRatio) > τ)
                updatedΡ = true;
            else
                updatedΡ = false;
            end
            if updatedΡ
                ρ = ρ * sqrt(resRatio);
                ρ = clamp.(ρ, T(1e-5), T(1e5));
                ρ¹  = inv(ρ);
                @. mK.nzval = mP.nzval + ρ * mAjAj.nzval;
                # sK = cholesky(Symmetric(mK); check = false);
                cholesky!(sK, Symmetric(mK); check = false);
            end
        end
    
    end

    vX[vJ] .= vXj;

    return vX, isConv;

end

function GenDiffOp( diffPow :: Int, numSamples :: Int )

    dCoeff = Dict(
        1 => [-0.5, 0.0, 0.5],
        2 => [1.0, -2.0, 1.0],
        3 => [-0.5, 1.0, 0.0, -1.0, 0.5],
        4 => [1.0, -4.0, 6.0, -4.0, 1.0],
        5 => [-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5],
        6 => [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0],
    )

    vC = dCoeff[diffPow];
    numCoeff = length(vC);
    coeffRadius = ((numCoeff - 1) ÷ 2);

    mD = spdiagm(numSamples, numSamples, (ii => vC[ii + coeffRadius + 1] * ones(numSamples - abs(ii)) for ii in -coeffRadius:coeffRadius)...);
    mD = mD[(coeffRadius + 1):(end - coeffRadius), :];

    return mD;

end

function GenMonoOp( vI :: Vector{Int}, vY :: Vector{T} ) where {T <: AbstractFloat}
    # Assumes `vI` is sorted

    numRefPts = length(vI);
    numSamples = length(vY);
    
    numSegments = numRefPts - 1;

    # Default Monotonic Non Decreasing
    # Forcing the monotonicity by the relationship with the next sample
    mD = ones(T, numSamples, 2); #<! Current sample
    mD[:, 2] .= T(-1); #<! Next sample

    for ii = 1:numSegments
        startIdx = vI[ii];
        endIdx   = vI[ii + 1];
        
        if vY[startIdx] <= vY[endIdx]
            # Monotonic Non Decreasing
            valSign = one(T);
        else
            # Monotonic Non Increasing
            valSign = -one(T);
        end
        
        for jj = startIdx:(endIdx - 1)
            mD[jj, :] .*= valSign;
        end
    
    end

    # https://discourse.julialang.org/t/39604
    mA = spdiagm(numSamples, numSamples, 0 => mD[:, 1], 1 => mD[1:(end - 1), 2]); 
    mA = mA[vI[1]:(vI[end] - 1), :];

    return mA;

end

## Parameters

# Data
csvFileName = "exchange_rate.csv"; #<! https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/exchange_rate
varIdx      = 2;
decFactor   = 50;

# Model
diffPow       = 2;
λ             = 1.95;
numRefPtsFctr = 0.025;


#%% Load / Generate Data

# Read the CSV Data
mData = readdlm(csvFileName, ',', skipstart = 1);
vY = Float64.(mData[:, varIdx]);
vY = vY[1:decFactor:end];

numSamples = length(vY);
vT = 1:numSamples;

mD = GenDiffOp(diffPow, numSamples);

numRefPts = Int(round(numRefPtsFctr * numSamples));
vI = sort(randperm(oRng, numSamples)[1:numRefPts]); #<! Not efficient

mP = I + λ * (mD' * mD);
vQ = -vY;

mA = sparse(1:numRefPts, vI, 1.0, numRefPts, numSamples); #<! Equality Constraints
vB = vY[vI];
mC = GenMonoOp(vI, vY); #<! Inequality Constraints
# mC.nzval .= 0.0; #<! Disable Monotonicity Constraints
vD = zeros(size(mC, 1));


## Analysis

# Reference Solution (QP Form)
vXRef = CVXSolver(mP, vQ, mA, vB, mC, vD);

# Optimized Solution
vX, isConv = SplineQPSmooth(vY, mD, λ, vI, mC);

println(isConv);
println(norm(vX - vXRef));


## Run Time Analysis
# @benchmark SplineQPSmoothCVX($vY, $mD, $λ, $vI, $mC)
# @benchmark SplineQPSmooth($vY, $mD, $λ, $vI, $mC)

## Display Results

figureIdx += 1;

sTr1 = scatter(; x = vT, y = vY, mode = "lines", 
               line_width = 3,
               name = "Data", text = "Data");
sTr2 = scatter(; x = vT[vI], y = vY[vI], mode = "markers", 
               marker_size = 10,
               name = "Reference Points", text = "Reference Points");
sTr3 = scatter(; x = vT, y = vX, mode = "lines", 
               line_width = 2,
               name = "Spline Curve", text = "Spline Curve");
sLayout = Layout(title = "Spline Piece Wise Monotonic Smooth", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "y",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));

hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end




