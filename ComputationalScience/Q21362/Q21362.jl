# StackExchange Computational Science Q21362
# https://scicomp.stackexchange.com/questions/21362
# Separate Text and Background in a Scanned Text Document.
# References:
#   1.  Signal Processing Q50329.
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
using ColorTypes;          #<! Required for Image Processing
using FileIO;              #<! Required for loading images
using Krylov;
using LinearOperators;
using LoopVectorization;   #<! Required for Image Processing
using PlotlyJS;
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

imgUrl = raw"https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Certificate_of_Arrival_for_Berta_Werner._-_NARA_-_282038.jpg/640px-Certificate_of_Arrival_for_Berta_Werner._-_NARA_-_282038.jpg";

# Problem parameters
numRows = 500; #<! Matrix K
numCols = numRows;  #<! Matrix K

λ = 0.05;
vλ = LinRange(0.0, 100.0, 8);


# Solver Parameters
numIterations   = Unsigned(7_500);
η               = 1e-6;
ρ               = 50.0;

#%% Load / Generate Data
mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = mI ./ 255.0;
vY = vec(mI);

numRows = size(mI, 1);
numCols = size(mI, 2);

mKₕ = [1.0 -1.0];      #<! Horizontal Derivative Kernel
mKᵥ = [[1.0, -1.0];;]; #<! Vertical Derivative Kernel

mDₕ = GenConvMtx(mKₕ, numRows, numCols; convMode = CONV_MODE_VALID);
mDᵥ = GenConvMtx(mKᵥ, numRows, numCols; convMode = CONV_MODE_VALID);


hObjFun( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = 0.5 * sum(abs2, vX .- 1.0) + (λ / 2.0) * sum(abs2, mDₕ * (vX - vY)) + (λ / 2.0) * sum(abs2, mDᵥ * (vX - vY));

dSolvers = Dict();


## Analysis
# The matrix: (I + λ * (mDₕ' * mDₕ + mDᵥ' * mDᵥ)) * x = b => A x = b

# Define the A operator

function OpAMul!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T, λ :: T, mO :: Matrix{T}, mB :: Matrix{T}, mV1 :: Matrix{T}, mV2 :: Matrix{T}, mKₕ :: Matrix{T}, mKᵥ :: Matrix{T} ) where {T <: AbstractFloat}

    # copyto!(mX, vX);
    # mX = reshape(vX, size(mO));
    mX = @invoke reshape(vX, size(mO));
    mO = CalcImageLaplacian!(mO, mX, mB, mV1, mV2, mKₕ, mKᵥ);
    if (β == zero(T))
        if (α == zero(T))
            vY .= zero(T);
        elseif (α == one(T))
            vY .= vX .+ λ .* mO[:];
        else
            vY .= α .* (vX .+ λ .* mO[:]);
        end
    elseif (β == one(T))
        if (α == one(T))
            vY .+= vX .+ λ .* mO[:];
        else
            vY .+= α .* (vX .+ λ .* mO[:]);
        end
    else
        if (α == zero(T))
            vY .*= β;
        elseif (α == one(T))
            vY .= (β .* vY) .+ vX .+ λ .* mO[:];
        else
            vY .= (β .* vY) .+ α .* (vX .+ λ .* mO[:]);
        end
    end

end

numRowsKₕ = size(mKₕ, 1);
numColsKₕ = size(mKₕ, 2);
numRowsmKᵥ = size(mKᵥ, 1);
numColsmKᵥ = size(mKᵥ, 2);

mO = similar(mI);
mB = similar(mI);
mV1 = Matrix{eltype(mI)}(undef, numRows - numRowsKₕ + 1, numCols - numColsKₕ + 1);
mV2 = Matrix{eltype(mI)}(undef, numRows - numRowsmKᵥ + 1, numCols - numColsmKᵥ + 1);

MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, λ, mO, mB, mV1, mV2, mKₕ, mKᵥ);

opA = LinearOperator(Float64, numRows * numCols, numRows * numCols, true, false, MulA!);
vB = 1.0 .+ λ .* (mDₕ' * mDₕ + mDᵥ' * mDᵥ) * vY;

oCgSolve = CgSolver(opA, vB);
cg!(oCgSolve, opA, vB);

vX = oCgSolve.x;
mX = similar(mI);
copyto!(mX, vX);


## Display Results

figureIdx += 1;

hP = DisplayImage(mX; titleStr = "λ = $λ");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

for (ii, λ) in enumerate(vλ)
    MulA!( vY :: AbstractVector{T}, vX :: AbstractVector{T}, α :: T, β :: T ) where {T <: AbstractFloat} = OpAMul!(vY, vX, α, β, λ, mO, mB, mV1, mV2, mKₕ, mKᵥ);
    local opA = LinearOperator(Float64, numRows * numCols, numRows * numCols, true, false, MulA!);
    local vB = 1.0 .+ λ .* (mDₕ' * mDₕ + mDᵥ' * mDᵥ) * vY;
    cg!(oCgSolve, opA, vB);
    copyto!(mX, oCgSolve.x);
    local hP = DisplayImage(mX; titleStr = "λ = $λ");
    display(hP);

    global figureIdx += 1;
    
    if (exportFigures)
        local figFileNme = @sprintf("Figure%04d.png", figureIdx);
        savefig(hP, figFileNme);
    end
end

# Run Time Analysis
runTime = @belapsed cg!(oCgSolve, opA, vB0) setup = (vB0 = copy(vB)) seconds = 2;
resAnalysis = @sprintf("The CG solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

mA = (I + λ * (mDₕ' * mDₕ + mDᵥ' * mDᵥ));
runTime = @belapsed (mA \ vB) seconds = 2;
resAnalysis = @sprintf("The Direct solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
