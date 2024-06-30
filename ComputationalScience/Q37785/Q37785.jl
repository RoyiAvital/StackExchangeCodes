# StackExchange Computational Science Q37785
# https://scicomp.stackexchange.com/questions/37785
# Solving Large Scale Sparse Linear System of Image Convolution.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     29/06/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Krylov;
using LinearOperators;
using LoopVectorization;   #<! Required for Image Processing
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


## Parameters

# Problem parameters
numRows = 500; #<! Matrix X
numCols = 450; #<! Matrix X

numRowsK = 09; #<! Matrix K
numColsK = 11; #<! Matrix K

λ = 0.05;

σ = 0.0; #<! Noise STD


# Solver Parameters


#%% Load / Generate Data

mX = rand(numRows, numCols);
mK = rand(numRowsK, numColsK);
vD = rand(numRows * numCols); #<! Image D (Not the Diagonal Matrix!!!)

mK⁻ = copy(mK); #<! mK Bar
tuPadRadius = (numRowsK ÷ 2, numColsK ÷ 2);
mK⁻[tuPadRadius[1] + 1, tuPadRadius[2] + 1] += λ;

mC = GenFilterMtx(mK, numRows, numCols; filterMode = FILTER_MODE_CONVOLUTION, boundaryMode = BND_MODE_CIRCULAR);
mI = λ * I;
mD = spdiagm(vD);

vY = (mC + mD + mI) * vec(mX) + (σ * randn(numRows * numCols));


## Analysis
# The matrix: (C + D + λ * I) * x = y => A x = y

# Define the A operator

function OpAMul!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T, mO :: Matrix{T}, tuPadRadius :: Tuple{N, N}, mP :: Matrix{T}, vD :: Vector{T}, mK :: Matrix{T} ) where {T <: AbstractFloat, N <: Signed}

    mX = @invoke reshape(vX, size(mO));
    mP = PadArray!(mP, mX, tuPadRadius; padMode = PAD_MODE_CIRCULAR);
    mO = Conv2D!(mO, mP, mK; convMode = CONV_MODE_VALID); #<! (C + λ * I) * x
    mO .+= (@invoke reshape(vD, size(mO))) .* mX; #<! (C + D + λ * I) * x (Could be optimized into the next calculation)
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

mO = similar(mX);
mP = zeros(eltype(mX), numRows + numRowsK - 1, numCols + numColsK - 1);

MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, mO, tuPadRadius, mP, vD, mK⁻);
MulAT!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, mO, tuPadRadius, mP, vD, rot180(mK⁻));

opA = LinearOperator(Float64, numRows * numCols, numRows * numCols, false, false, MulA!, MulAT!);

oKrylovSolver = CglsSolver(opA, vY);
cgls!(oKrylovSolver, opA, vY);
println(norm(oKrylovSolver.x - mX[:], Inf) * 255.0);
println(@belapsed cgls!(oKrylovSolver, opA, vB) setup = (vB = copy(vY)) seconds = 2);

oKrylovSolver = LsmrSolver(opA, vY);
lsmr!(oKrylovSolver, opA, vY);
println(norm(oKrylovSolver.x - mX[:], Inf) * 255.0);
println(@belapsed lsmr!(oKrylovSolver, opA, vB) setup = (vB = copy(vY)) seconds = 2);

oKrylovSolver = UsymqrSolver(opA, vY);
usymqr!(oKrylovSolver, opA, vY, copy(vY));
println(norm(oKrylovSolver.x - mX[:], Inf) * 255.0);
println(@belapsed usymqr!(oKrylovSolver, opA, vB, copy(vY)) setup = (vB = copy(vY)) seconds = 2);

oKrylovSolver = LsqrSolver(opA, vY);
lsqr!(oKrylovSolver, opA, vY);
println(norm(oKrylovSolver.x - mX[:], Inf) * 255.0);
println(@belapsed lsqr!(oKrylovSolver, opA, vB) setup = (vB = copy(vY)) seconds = 2);



# vX = oCglsSolver.x;


## Display Results

# figureIdx += 1;

# hP = DisplayImage(mX; titleStr = "λ = $λ");
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end

# figureIdx += 1;

# hP = DisplayImage(mI; titleStr = "Input Image");
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end

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
