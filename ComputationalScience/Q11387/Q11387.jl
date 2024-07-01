# StackExchange Computational Science Q11387
# https://scicomp.stackexchange.com/questions/311387
# Interpolation by Solving a Minimization Problem (Optimization).
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  Working with 4 Connectivity seems to be better than 8 Connectivity.
# TODO:
# 	1.  Use `Krylov.jl` to support larger matrices.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     01/07/2024  Royi Avital
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

function BuildImgGraph( mI :: Matrix{T}, hW :: Function, winRadius :: N ) where {T <: AbstractFloat, N <: Integer}
    # Build a graph of LxL neighborhood using the weights function.
    # The function must return a tuple of the value and if the value is valid (To be inserted into the adjacency graph).
    # It might be useful to only calculate the distance between the neighborhood.  
    # Then one can apply per row normalization and global normalization then apply element wise weighing.

    numRows = size(mI, 1);
    numCols = size(mI, 2);
    numPx   = numRows * numCols;
    winLen  = (N(2) * winRadius) + one(N);

    # Number of edges (Ceiled estimation as on edges there are less)
    vI = ones(Int32, winLen * winLen * numPx); #<! Must be valid index
    vJ = ones(Int32, winLen * winLen * numPx); #<! Must be valid index
    vV = zeros(T, winLen * winLen * numPx); #<! Add zero value

    elmIdx = 0;
    refPxIdx = 0;
    for jj ∈ 1:numCols
        for ii ∈ 1:numRows
            refPxIdx += 1;
            for nn ∈ -winRadius:winRadius
                for mm ∈ -winRadius:winRadius

                    if (((ii + mm) > 0) && ((ii + mm) <= numRows) && ((jj + nn) > 0) && ((jj + nn) <= numCols))
                        # Pair is within neighborhood
                        weightVal, isValid = hW(mI[ii, jj], mI[ii + mm, jj + nn], ii, jj, mm, nn);
                        if (isValid)
                            elmIdx += 1;
                            vI[elmIdx] = refPxIdx;
                            vJ[elmIdx] = refPxIdx + (nn * numRows) + mm;
                            vV[elmIdx] = weightVal
                        end
                    end
                end
            end
        end
    end

    mW = sparse(vI[1:elmIdx], vJ[1:elmIdx], vV[1:elmIdx], numPx, numPx);

    return mW;

end


## Parameters

imgUrlGray   = raw"https://i.sstatic.net/gjTJa.png";
imgUrlMarked = raw"https://i.sstatic.net/0oqlt.png";

# Problem parameters

# 8 Connectivity
hW(valI :: T, valN :: T, ii :: N, jj :: N, mm :: N, nn :: N) where {T <: AbstractFloat, N <: Integer} = (abs(valI - valN), true); #<! Weighing function
# 4 Connectivity (Excludes center)
hW(valI :: T, valN :: T, ii :: N, jj :: N, mm :: N, nn :: N) where {T <: AbstractFloat, N <: Integer} = ifelse(((mm * nn == zero(N)) && ((mm !=- zero(N)) || (nn !=- zero(N)))), (abs(valI - valN), true), (zero(T), false)); #<! Weighing function

τ         = 0.25;
ϵ         = 1e-5;
mC        = GenColorConversionMat(RGB_TO_YIQ); #<! Color conversion matrix
winRadius = 1;
β         = 200.0;

# Solver Parameters


#%% Load / Generate Data

# Gray / Original Image
mI = load(download(imgUrlGray));
mI = ConvertJuliaImgArray(mI);
mI = mI ./ 255.0;

# Marked Image
mM = load(download(imgUrlMarked));
mM = ConvertJuliaImgArray(mM);
mM = mM ./ 255.0;

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx   = numRows * numCols;


## Analysis

mMYiq = ConvertColorSpace(mM, mC);
mOYiq = ConvertColorSpace(mI, mC); #<! Check if needed

mB = sum(abs.(mI .- mM), dims = 3) .> τ;
vV = findall(mB[:]); #<! Indices of marks (Set \mathcal{V})

# Distance Matrix (Graph)
mW = BuildImgGraph(mI[:, :, 1], hW, winRadius);
# Scale DR linearly
minVal = minimum(mW.nzval);
maxVal = maximum(mW.nzval);
mW.nzval .= (mW.nzval .- minVal) ./ (maxVal - minVal); 

mW.nzval .= exp.(-β .* mW.nzval) .+ ϵ; #<! Distance -> Weights
vD = vec(sum(mW, dims = 2));
mD = spdiagm(0 => vD); #<! Degree Matrix (Diagonal of the sum of each row)
mL = mD .- mW;

vU = setdiff(1:numPx, vV); #<! Rest of unlabeled pixels (Set \mathcal{U})

mLᵤ = mL[vU, vU]; #<! The Laplacian sub matrix to optimize by
mR  = mL[vU, vV];
oFLᵤ = cholesky(mLᵤ); #<! Symbolic factorization, supports in place (https://discourse.julialang.org/t/6091).

# Solving (Per channel): Lᵤ xᵤ = −R d
for ii ∈ 1:2
    mChn = view(mMYiq, :, :, ii + 1);
    vXᵥ  = mChn[vV]; #<! Anchor values
    vXᵤ = -(oFLᵤ \ (mR * vXᵥ));
    mChn = view(mOYiq, :, :, ii + 1);
    mChn[vV] = vXᵥ;
    mChn[vU] = vXᵤ;
end

mO = ConvertColorSpace(mOYiq, inv(mC));


## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mM; titleStr = "Marker Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = PlotSparseMat(mW); #<! Too large to display

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.html", figureIdx);
#     savefig(hP, figFileNme);
# end

