# StackExchange Mathematics Q4993451
# https://math.stackexchange.com/questions/4993451
# Apply Max Lloyd Quantizer on a Synthetic 2D Image.
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
# - 1.0.000     05/11/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
using StatsBase;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function GetImageHistCount( mI :: Matrix{T}  ) where {T <: AbstractFloat}

    vX = unique(mI);
    vC = [count(==(valI), mI) for valI in vX]; #<! Count
    
    # Optional (Less Memory, not N^2)
    # dPdf = Dict{T, Int64}();
    # for valI in mI
    #     dPdf[valI] = get(dPdf, valI, 0) + 1;
    # end
    # vX = collect(keys(dPdf));
    # vC = collect(values(dPdf));
    # vI = sortperm(vX);
    # vX = vX[vI];
    # vC = vC[vI];

    return vX, vC;

end

function Expect( hF :: Function, vX :: Vector{T}, vF :: Vector{T}, lowerBnd :: T, upperBnd :: T ) where {T <: AbstractFloat}
    # Replication of SciPy `expect()` for discrete distribution

    valE = zero(T);
    for (valX, valF) in zip(vX, vF)
        valE += ifelse(valX <= upperBnd && valX >= lowerBnd, valF * hF(valX), zero(T));
    end

    return valE;
    
end

function QunatizeImage( mI :: Matrix{T}, vT :: Vector{T}, vR :: Vector{T} ) where {T <: AbstractFloat}

    numRows     = size(mI, 1);
    numCols     = size(mI, 2);
    
    mO = copy(mI);

    # Inefficient
    for jj ∈ 1:numCols, ii ∈ 1:numRows
        # lvlIdx = max(findfirst(x -> x >= (mO[ii, jj]), vT) - 1, 1);
        lvlIdx = max(findfirst(>=(mO[ii, jj]), vT) - 1, 1);
        mO[ii, jj] = vR[lvlIdx];
    end

    return mO;    

end


## Parameters

tuGridBnd  = (0.0, 10.0);
numGridPts = 101;

numLevels = 16; #<! 2^8
numIter   = 10;


## Load / Generate Data

hI( xx :: T, yy :: T ) where {T <: AbstractFloat} = (xx ^ 2) + (yy ^ 2); #<! Image function
vG = LinRange(tuGridBnd[1], tuGridBnd[2], numGridPts);
mI = [hI(xx, yy) for yy in vG, xx in vG]; #<! The first `for` iterates first (https://discourse.julialang.org/t/75659)

## Analysis

# Building the Histogram
vX, vC = GetImageHistCount(mI);

vF = vC / sum(vC); #<! Probability

vT = collect(LinRange(minimum(mI), maximum(mI), numLevels + 1));
vR = zeros(numLevels);

hI( valX :: T ) where {T <: AbstractFloat} = valX;
hC( valX :: T ) where {T <: AbstractFloat} = one(T);

for ii ∈ 1:numIter
    for jj ∈ 1:numLevels
        vR[jj] = Expect(hI, vX, vF, vT[jj], vT[jj + 1]) / Expect(hC, vX, vF, vT[jj], vT[jj + 1]);
    end
    for jj ∈ 2:numLevels
        vT[jj] = 0.5 * (vR[jj - 1] + vR[jj]);
    end
end

mO = QunatizeImage(mI, vT, vR);


## Display Analysis

figureIdx += 1;

oTr = heatmap(; x = vG, y = vG, z = mI, zmin = vX[1], zmax = vX[end]);
oLayout = Layout(title = "Input Image", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

# Quantize the image to make the histogram viewable
vXX, vCC = GetImageHistCount(round.(mI * 10.0) ./ 10.0);

oTr = bar(; x = vXX, y = vCC, marker_color = "red");
oLayout = Layout(title = "Input Image PDF", width = 800, height = 600, 
                 xaxis_title = "Image Value", yaxis_title = "Probability",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr = heatmap(; x = vG, y = vG, z = mO, zmin = vX[1], zmax = vX[end]);
oLayout = Layout(title = "Qunatized Image", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


