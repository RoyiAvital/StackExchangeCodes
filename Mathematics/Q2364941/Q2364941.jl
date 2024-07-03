# StackExchange Mathematics Q2364941
# https://math.stackexchange.com/questions/4938099
# Equalizing Image Histogram.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     03/07/2024  Royi Avital
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

function CalcImgHist( mI :: Matrix{T} ) where{T <: Unsigned}

    numValues = typemax(T) + 1;
    vH = zeros(UInt32, numValues);

    for ii ∈ eachindex(mI)
        vH[mI[ii] + 1] += one(UInt32);
    end

    return vH;

end

function EqualizeImg( mI :: Matrix{T} ) where{T <: Unsigned}

    vH = CalcImgHist(mI);
    vC = cumsum(vH);
    mO = similar(mI);

    numPx = length(mI);
    maxVal = typemax(T);

    for ii ∈ eachindex(mI)
        mO[ii] = T(floor(maxVal * (vC[mI[ii] + UInt32(1)] / numPx)));
    end

    return mO;

end


## Parameters

imgUrl = raw"https://i.sstatic.net/MMWYH.png"; #<! Image of the question
imgUrl = raw"https://upload.wikimedia.org/wikipedia/commons/0/08/Unequalized_Hawkes_Bay_NZ.jpg";


#%% Load / Generate Data

mI = load(download(imgUrl));
mI = ConvertJuliaImgArray(mI); #<! Keeps it UInt
mI = mI[:, :, 1]; #<! Image is gray scale


## Analysis

imgEltType = eltype(mI);
vHᵢ = CalcImgHist(mI);
vCᵢ = cumsum(vHᵢ);
mO = EqualizeImg(mI);
vHₒ = CalcImgHist(mO);
vCₒ = cumsum(vHₒ);


## Display Results

figureIdx += 1;

oTr = bar(x = 0:typemax(imgEltType), y = vHᵢ, name = "Input Image",
                line = attr(width = 3.0));
oLayout = Layout(title = "Image Histogram - Input Image", width = 900, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Count [Px]",
                 bargap = 0.0);
hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr = bar(x = 0:typemax(imgEltType), y = vHₒ, name = "Output Image",
                line = attr(width = 3.0));
oLayout = Layout(title = "Image Histogram - Output Image", width = 900, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Count [Px]",
                 bargap = 0.0);
hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr = bar(x = 0:typemax(imgEltType), y = vCᵢ, name = "Input Image",
                line = attr(width = 3.0));
oLayout = Layout(title = "Image CDF - Input Image", width = 900, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Count [Px]",
                 bargap = 0.0);
hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr = bar(x = 0:typemax(imgEltType), y = vCₒ, name = "Output Image",
                line = attr(width = 3.0));
oLayout = Layout(title = "Image CDF - Output Image", width = 900, height = 600, hovermode = "closest",
                 xaxis_title = "Value", yaxis_title = "Count [Px]",
                 bargap = 0.0);
hP = plot([oTr], oLayout);
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

figureIdx += 1;

hP = DisplayImage(mO; titleStr = "Output Image");
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
