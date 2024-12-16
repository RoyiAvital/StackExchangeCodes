# StackExchange Signal Processing Q95071
# https://dsp.stackexchange.com/questions/95071
# Build the Laplacian Matrix of Edge Preserving Multiscale Image Decomposition based on Local Extrema.
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
# - 1.0.000     17/09/2024  Royi Avital
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
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StaticKernels;       #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images
include(joinpath(juliaCodePath, "JuliaSparseArrays.jl")); #<! Sparse Arrays

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function BuildImgGraph( mI :: Matrix{T}, hV :: Function, hW :: Function, winRadius :: N ) where {T <: AbstractFloat, N <: Integer}
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

    elmIdx   = 0;
    refPxIdx = 0;
    for jj ∈ 1:numCols, ii ∈ 1:numRows
        refPxIdx += 1;
        for nn ∈ -winRadius:winRadius, mm ∈ -winRadius:winRadius
            if (((ii + mm) > 0) && ((ii + mm) <= numRows) && ((jj + nn) > 0) && ((jj + nn) <= numCols))
                # @infiltrate ((ii == 3) && (jj == 1))
                # Pair is within neighborhood
                isValid = hV(ii, jj, mm, nn);
                if (isValid)
                    elmIdx    += 1;
                    weightVal  = hW(mI[ii, jj], mI[ii + mm, jj + nn], ii, jj, mm, nn);
                    vI[elmIdx] = refPxIdx;
                    vJ[elmIdx] = refPxIdx + (nn * numRows) + mm;
                    vV[elmIdx] = weightVal;
                end
            end
        end
    end

    # Julia does include explicit zeros. Hence the mapping to (1, 1) 
    # will create an item at `mW[1, 1]` which might cause issues.
    mW = sparse(vI[1:elmIdx], vJ[1:elmIdx], vV[1:elmIdx], numPx, numPx);

    return mW;

end

function LocalSparseInterpolation( mI :: Matrix{T}, mM :: BitMatrix, winRadius :: N; ϵ :: T = T(1e-5) ) where {T <: AbstractFloat, N <: Integer}

    # Validation Function
    hV(ii :: N, jj :: N, mm :: N, nn :: N) where {N <: Integer} = (abs(mm) <= N(1)) && (abs(nn) <= N(1)) && ((mm != zero(N)) || (nn != zero(N))); #<! 8 Connectivity
    # hV(ii :: N, jj :: N, mm :: N, nn :: N) where {N <: Integer} = (mm * nn == zero(N)) && ((mm != zero(N)) || (nn != zero(N))); #<! 4 Connectivity
    
    # Weighing Function
    hW(valI :: T, valN :: T, ii :: N, jj :: N, mm :: N, nn :: N) where {T <: AbstractFloat, N <: Integer} = abs(valI - valN) + ϵ; #<! Weighing function
    # hW(valI :: T, valN :: T, ii :: N, jj :: N, mm :: N, nn :: N) where {T <: AbstractFloat, N <: Integer} = exp(-((valI - valN) ^ 2) / (T(2) * mV[ii, jj])); #<! Weighing function
    
    # Distance Matrix (Graph)
    # Number of non zeros for 8 connectivity: 8 * (numRows - 2) * (numCols - 2) + 10 * (numRows - 2) + 10 * (numCols - 2) + 4 * 3
    mW = BuildImgGraph(mI, hV, hW, winRadius);

    # Local Variance Image
    mK = Kernel{(-winRadius:winRadius, -winRadius:winRadius)}(@inline w -> var(Tuple(w)));
    mV = map(mK, extend(mI, StaticKernels.ExtensionSymmetric()));

    # Local Smoothness
    mK = Kernel{(-winRadius:winRadius, -winRadius:winRadius)}(@inline w -> minimum(((w[-1, -1], w[-1, 0], w[-1, 1], w[0, -1], w[0, 1], w[1, -1], w[1, 0], w[1, 1]) .- w[0, 0]) .^ 2));
    mGV = map(mK, extend(mI, StaticKernels.ExtensionSymmetric()));

    # Affinity Matrix Distance -> Weights
    vR, vC, vVals = findnz(mW);
    for ii ∈ 1:length(vR)
        localVar  = 0.6 * mV[vR[ii]]; #<! The row is the reference pixel index
        mgVal     = mGV[vR[ii]];
        localVar  = max(localVar, -mgVal / log(0.01));
        localVar  = max(localVar, 0.000002) / 2;
        vVals[ii] = exp(-(vVals[ii] * vVals[ii]) / (2 * localVar)); #<! Exponent function
    end
    
    mW = sparse(vR, vC, vVals, numPx, numPx);
    # Rows of Sum 1
    mW = NormalizeRows(mW);
    
    vV = findall(mM[:]); #<! Indices of marks (Set \mathcal{V})

    # Pseudo Laplacian Matrix (As it is neither symmetric nor PD)
    vD = vec(sum(mW; dims = 2));
    mD = spdiagm(0 => vD); #<! Degree Matrix (Diagonal of the sum of each row)
    mL = mD .- mW; #<! Pseudo Laplacian Matrix

    # Permutation Matrix like effect by selection
    vU = setdiff(1:numPx, vV); #<! Rest of unlabeled pixels (Set \mathcal{U})
    
    # Permutation of **Rows and Columns** of SPD matrix will yield SPD matrix (https://math.stackexchange.com/questions/3559710)
    # Actually permutation is not needed, as the choice of indices in the graph is arbitrary.
    # Hence the system can be shown as:
    # [ mLu, mLv ] * [ vXu ] = [  0  ]
    # [  0 ,  I  ] * [ vXv ] = [ vXv ]
    # The first row means: mLu * vXu + mLv * vXv = 0 => mLu * vXu = -mLv * vXv.
    # Hence one can solve: vXu = mLu \ (-mLv * vXv).
    # Here `mR` is `mLv`.
    mLᵤ = mL[vU, vU]; #<! The Laplacian sub matrix to optimize by
    mR  = mL[vU, vV];

    vXᵥ  = mI[vV]; #<! Anchor values
    vXᵤ = -(mLᵤ \ (mR * vXᵥ));

    mO = similar(mI);
    mO[vV] = vXᵥ;
    mO[vU] = vXᵤ;

    return mO;

end

function LocalExtremaSmoothing( mI :: Matrix{T}, k :: N ) where {T <: AbstractFloat, N <: Integer}

    localLen = 2k + 1; #<! k in the paper

    mLocalKValue = OrderFilter(mI, k, localLen * localLen - localLen + 1);
    mLocalMax = mI .>= mLocalKValue; #<! Local Maximum
    mLocalKValue = OrderFilter(mI, k, localLen);
    mLocalMin = mI .<= mLocalKValue; #<! Local Minimum

    mOMax = LocalSparseInterpolation(mI, mLocalMax, k);
    mOMin = LocalSparseInterpolation(mI, mLocalMin, k);

    return 0.5 * (mOMax + mOMin);

end


## Parameters

imgUrl = raw"https://i.postimg.cc/85Jjs9wJ/Flowers.png"; #<! https://i.imgur.com/PckT6jF.png
imgUrl = raw"https://raw.githubusercontent.com/yafangshih/EdgePreserving-Blur/refs/heads/master/data/input/taipei101.jpg"; #<! For timing

# Problem parameters

paramK = 2; #<! Radius (The K in the paper K = 2k + 1)


#%% Load / Generate Data

# Gray / Original Image
mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI);
mI = ScaleImg(mI);
if (ndims(mI) == 3)
    mI = mean(mI; dims = 3);
    mI = dropdims(mI; dims = 3);
end

numRows = size(mI, 1);
numCols = size(mI, 2);
numPx   = numRows * numCols;


## Analysis

mO = LocalExtremaSmoothing(mI, paramK);


## Display Results

figureIdx += 1;

hP = DisplayImage(mI; titleStr = "Input Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image, k = $(paramK)");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
