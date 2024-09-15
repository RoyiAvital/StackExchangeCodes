# StackOverflow Q12300628
# https://stackoverflow.com/questions/12300628
# Generating and Solving the Sparse Linear System of Poisson Image Editing.
# References:
#   1.  Christopher J. Tralie - Poisson Image Editing - http://www.ctralie.com/Teaching/PoissonImageEditing.
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     14/09/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using BenchmarkTools;
using ColorTypes;        #<! Required for Image Processing
using PlotlyJS;          #<! Use `add Kaleido_jll@v0.1` (https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using FileIO;            #<! Required for loading images
using LoopVectorization; #<! Required for Image Processing
using StableRNGs;
using StaticKernels;     #<! Required for Image Processing


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

@enum OpMode begin
    OP_MODE_AVERAGE
    OP_MODE_MAXIMUM
    OP_MODE_REPLACE
end

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function BuildModelData( mM :: BitMatrix, mS :: Matrix{T}, mT :: Matrix{T} ) where {T <: AbstractFloat}
    # Though intuitively one would create a zero sum row Laplacian Matrix it should not.
    # In this case the classic zero sum rows Laplacian will yield bad result.  
    # So, if the mask is not on the edge of the image, all pixels has 4 neighbors.  
    # Even if some neighbors are not part of the mask as long as they are part of the image.

    numRows = size(mM, 1);
    numCols = size(mM, 2);
    
    # Maps (ii, jj) -> Mask Pixel Index (Columns stack)
    mMap = zeros(UInt, numRows, numCols);
    kk = 0;
    for jj ∈ 2:(numCols - 1), ii ∈ 2:(numRows - 1)
        # Assuming the mask does not touch the boundaries
        if mM[ii, jj]
            kk += 1;
            mMap[ii, jj] = kk;
        end
    end
    
    numPxMask   = sum(mM);
    vG          = zeros(T, numPxMask);
    # 5 Points system (At most 5 per row of `mC`)
    vI          = ones(Int, 5 * numPxMask);
    vJ          = ones(Int, 5 * numPxMask);
    vV          = zeros(T, 5 * numPxMask);

    kk = 0; #<! Element Index (In `vI`, `vJ`, `vV`)
    rr = 0; #<! Row Counter (In `mC`)
    for jj ∈ 2:(numCols - 1), ii ∈ 2:(numRows - 1)
        # Assuming the mask does not touch the boundaries
        if mM[ii, jj]
            rr += 1; #<! The row in `mC`
            kk += 1; #<! Element index in `vI`, `vJ`, `vV`
            ll  = kk; #<! To access the current diagonal of `mC`
            vI[kk] = rr;
            vJ[kk] = rr;
            # vV[kk] = zero(T); #<! Redundant (For clarity)
            vV[kk] = T(4.0); #<! Must, as the Laplacian Matrix is singular
            vG[rr] = -(-T(4.0) * mS[ii, jj] + mS[ii - 1, jj] + mS[ii + 1, jj] + mS[ii, jj - 1] + mS[ii, jj + 1]);
            if mM[ii - 1, jj]
                # vV[ll] += one(T);
                kk     += 1; #<! New element on the row
                vI[kk]  = rr;
                vJ[kk]  = mMap[ii - 1, jj]; #<! Column index
                vV[kk]  = -one(T);
            else
                vG[rr] += mT[ii - 1, jj]
            end
            if mM[ii + 1, jj]
                # vV[ll] += one(T);
                kk     += 1; #<! New element on the row
                vI[kk]  = rr;
                vJ[kk]  = mMap[ii + 1, jj]; #<! Column index
                vV[kk]  = -one(T);
            else
                vG[rr] += mT[ii + 1, jj]
            end
            if mM[ii, jj - 1]
                # vV[ll] += one(T);
                kk     += 1; #<! New element on the row
                vI[kk]  = rr;
                vJ[kk]  = mMap[ii, jj - 1]; #<! Column index
                vV[kk]  = -one(T);
            else
                vG[rr] += mT[ii, jj - 1]
            end
            if mM[ii, jj + 1]
                # vV[ll] += one(T);
                kk     += 1; #<! New element on the row
                vI[kk]  = rr;
                vJ[kk]  = mMap[ii, jj + 1]; #<! Column index
                vV[kk]  = -one(T);
            else
                vG[rr] += mT[ii, jj + 1]
            end
        end
    end

    # Zeros are mapped to `mC[1, 1]`.
    # By default recurrent indices are additive, hence no zeros to drop.
    mC = sparse(vI, vJ, vV, numPxMask, numPxMask);
    
    return mC, vG;
    
end


## Parameters

# Data
# Source: https://cw.fel.cvut.cz/b241/courses/b4m33dzo/labs/5_poisson
maskUrl = raw"https://i.imgur.com/jxrIdoI.png"; #<! Alternative https://i.postimg.cc/D0nmkqC0/Mask.png
srcUrl  = raw"https://i.imgur.com/PrInOGl.png"; #<! Alternative https://i.postimg.cc/h41f85jd/Source.png
tgtUrl  = raw"https://i.imgur.com/9ND8Lhv.png"; #<! Alternative https://i.postimg.cc/BbSZwJ61/Target.png

# Model
opMode = OP_MODE_REPLACE;

# Solver Parameters
numIter = 100; #<! Iterations


#%% Load / Generate Data

mM = load(download(maskUrl)); #<! Mask
mM = ConvertJuliaImgArray(mM);
mM = ScaleImg(mM);
mM = mM[:, :, 1];
mM = Bool.(mM); #<! May use `collect()` for `Matrix{Bool}`
mS = load(download(srcUrl)); #<! Source
mS = ConvertJuliaImgArray(mS);
mS = ScaleImg(mS);
mT = load(download(tgtUrl)); #<! Target
mT = ConvertJuliaImgArray(mT);
mT = ScaleImg(mT);

numRows = size(mM, 1);
numCols = size(mM, 2);

numPx = numRows * numCols;


# ## Analysis

mH = copy(mT); #<! Initialize by the target

for ii ∈ 1:3
    mC, vG = BuildModelData(mM, mS[:, :, ii], mT[:, :, ii]);
    vH = mC \ vG;
    mH[mM, ii] = vH; #<! Works in Julia, Does not in MATLAB
end


# ## Display Results

figureIdx += 1;

hP = DisplayImage(mS; titleStr = "Source Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

hP = DisplayImage(mT; titleStr = "Target Image");
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

hP = DisplayImage(mH; titleStr = "Fused Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end