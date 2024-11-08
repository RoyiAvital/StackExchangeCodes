# StackExchange Computational Science Q44550
# https://scicomp.stackexchange.com/questions/44550
# Solve Large Scale Underdetermined Linear Equation with per Element Equality Constraint.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  Use `Krylov.jl` to support larger matrices.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     21/09/2024  Royi Avital
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
using LoopVectorization;   #<! Required for Image Processing
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaSparseArrays.jl")); #<! Arrays
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function BuildImgGraphAE( mI :: Matrix{T}, mM :: BitMatrix, connMode :: ConnMode; ε :: T = T(1e-5) ) where {T <: AbstractFloat}
    # Build a graph of 3x3 neighborhood.
    # Diagonal is 1.
    # For non mask pixels the off diagonal are exponentially weighted.
    # Based on the graph model in Colorization Using Optimization.

    numRows = size(mI, 1);
    numCols = size(mI, 2);
    numPx   = numRows * numCols;
    winLen  = 3;

    # Number of edges (Ceiled estimation as on edges there are less)
    vI = ones(Int32, winLen * winLen * numPx); #<! Must be valid index
    vJ = ones(Int32, winLen * winLen * numPx); #<! Must be valid index
    vV = zeros(T, winLen * winLen * numPx); #<! Add zero value

    # Vector of the local values
    vL = zeros(T, 9); #<! Connectivity 4: 5, Connectivity 8: 9

    elmIdx   = 0;
    refPxIdx = 0;
    for jj ∈ 1:numCols, ii ∈ 1:numRows
        refPxIdx += 1;
        if (mM[ii, jj])
            # Pixel in V (The mask)
            elmIdx += 1;
            vI[elmIdx] = refPxIdx;
            vJ[elmIdx] = refPxIdx;
            vV[elmIdx] = one(T);
        else
            # Pixel in U
            localPxIdx = 0;
            for nn ∈ -1:1, mm ∈ -1:1
                if (((ii + mm) > 0) && ((ii + mm) <= numRows) && ((jj + nn) > 0) && ((jj + nn) <= numCols)) && ((mm != 0) || (nn != 0))
                    localPxIdx += 1;
                    elmIdx     += 1;

                    vI[elmIdx]      = refPxIdx;
                    vJ[elmIdx]      = refPxIdx + (nn * numRows) + mm;
                    vL[localPxIdx]  = mI[ii + mm, jj + nn];
                end
            end
            ll = localPxIdx + 1;
            vL[ll] = mI[ii, jj];
            σ² = var(@view(vL[1:ll])) + ε;
            for kk ∈ 1:localPxIdx
                vL[kk] = exp(-((vL[kk] - vL[ll]) ^ 2) / (σ²));
            end
            # Normalizing the sum to 1
            sumVal = sum(@view(vL[1:localPxIdx]));
            for (ll, kk) ∈ enumerate((elmIdx - localPxIdx + 1):elmIdx)
                vV[kk] = -vL[ll] / sumVal; #<! Minus Value as the sum with the diagonal is 0
            end
            # Diagonal element
            elmIdx += 1;
            vI[elmIdx] = refPxIdx;
            vJ[elmIdx] = refPxIdx;
            vV[elmIdx] = one(T);
        end
    end

    # Julia does include explicit zeros. Hence the mapping to (1, 1) 
    # will create an item at `mW[1, 1]` which might cause issues.
    mW = sparse(vI[1:elmIdx], vJ[1:elmIdx], vV[1:elmIdx], numPx, numPx);

    return mW;

end

function SolveBSystem( mA :: AbstractSparseMatrix{T}, mI :: Matrix{T}, mM :: BitMatrix ) where {T <: AbstractFloat}

    vM = vec(mM);
    mB = mA' * mA;
    # Partitioning by U, V
    mBu = mB[.!vM, .!vM];
    mC  = mB[.!vM, vM];

    vXv = mI[vM];
    
    vXu, sSol = cg(mBu, -mC * vXv);

    vX = zeros(T, length(mI));
    vX[vM] = vXv;
    vX[.!vM] = vXu;

    return vX;

end

function SolveAESystem( mAE :: AbstractSparseMatrix{T}, mI :: Matrix{T}, mM :: BitMatrix ) where {T <: AbstractFloat}

    vM = vec(mM);
    vB = zeros(T, length(mI));

    vB[vM] = mI[vM];
    
    vX, sSol = cgls(mAE, vB);

    return vX;

end

function SolveAESystemRef( mAE :: AbstractSparseMatrix{T}, mI :: Matrix{T}, mM :: BitMatrix ) where {T <: AbstractFloat}

    vM = vec(mM);
    vB = zeros(T, length(mI));

    vB[vM] = mI[vM];
    vX = mAE \ vB;

    return vX;

end


## Parameters

imgUrl  = raw"https://i.imgur.com/yhb5xdW.png"; #<! https://i.postimg.cc/5tPvGCzd/Img.png
maskUrl = raw"https://i.imgur.com/Idk19pd.png"; #<! https://i.postimg.cc/7ZnS3cYL/Color-Mask.png

# Problem parameters
ε = 1e-5;
connMode = CONN_MODE_4;


#%% Load / Generate Data

# Gray / Original Image
mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = ScaleImg(mI);

mM = load(download(maskUrl));
mM = ConvertJuliaImgArray(mM);
mM = (mM[:, :, 1] .!= mM[:, :, 2]) .|| (mM[:, :, 1] .!= mM[:, :, 3]);

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx   = numRows * numCols;


## Analysis

mAE = BuildImgGraphAE(mI, mM, connMode; ε = ε);
mA  = mAE[.!mM[:], :];

vXR = SolveAESystemRef(mAE, mI, mM);

vXA = SolveAESystem(mAE, mI, mM);
vXB = SolveBSystem(mA, mI, mM);

# Reshape to image
mXA = collect(reshape(vXA, (numRows, numCols)));
mXB = collect(reshape(vXB, (numRows, numCols)));

maxAbsDev = maximum(abs.(vXA - vXR));
resAnalysis = @sprintf("Direct <-> Reference Maximum Absolute Deviation: %0.12f", maxAbsDev);
println(resAnalysis);

maxAbsDev = maximum(abs.(vXB - vXR));
resAnalysis = @sprintf("LS <-> Reference Maximum Absolute Deviation: %0.12f", maxAbsDev);
println(resAnalysis);

# Timing Operators
runTime = @belapsed SolveAESystem($(mAE), $(mI), $(mM)) seconds = 2;
resAnalysis = @sprintf("Direct Solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

runTime = @belapsed SolveBSystem($(mA), $(mI), $(mM)) seconds = 2;
resAnalysis = @sprintf("LS Solution run time: %0.5f [Sec]", runTime);
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

hP = DisplayImage(mM; titleStr = "Mask Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mXB; titleStr = "Direct Solution");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mXA; titleStr = "LS Solution");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
