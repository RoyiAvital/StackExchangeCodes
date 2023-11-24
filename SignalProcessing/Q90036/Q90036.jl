# StackExchange Signal Processing (DSP) Q90036
# https://dsp.stackexchange.com/questions/90036
# Use DFT (`fft()`) to Replicate 2D Convolution `(conv())`.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
#   3. 
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     24/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using FFTW;
using FileIO;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaImageProcessing.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions

function Conv2DFreqDom( mI :: Matrix{T}, mH :: Matrix{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat}

    numRowsI, numColsI = size(mI);
    numRowsH, numColsH = size(mH);

    if (convMode == CONV_MODE_FULL)
        numRowsFft  = numRowsI + numRowsH - 1;
        numColsFft  = numColsI + numColsH - 1;
        mO = Matrix{T}(undef, (numRowsI, numColsI) .+ (numRowsH, numColsH) .- 1);
    elseif (convMode == CONV_MODE_SAME)
        numRowsFft  = numRowsI + numRowsH;
        numColsFft  = numColsI + numColsH;
        mO = Matrix{T}(undef, (numRowsI, numColsI));
    elseif (convMode == CONV_MODE_VALID)
        numRowsFft  = numRowsI;
        numColsFft  = numColsI;
        mO = Matrix{T}(undef, (numRowsI, numColsI) .- (numRowsH, numColsH) .+ 1);
    end

    mT1 = Matrix{complex(T)}(undef, numRowsFft, numColsFft);
    mT2 = Matrix{complex(T)}(undef, numRowsFft, numColsFft);

    Conv2DFreqDom!(mO, mI, mH, mT1, mT2; convMode = convMode);

    return mO;

end

function Conv2DFreqDom!( mO :: Matrix{T}, mI :: Matrix{T}, mH :: Matrix{T}, mT1 :: Matrix{Complex{T}}, mT2 :: Matrix{Complex{T}}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat}

    numRowsI, numColsI = size(mI);
    numRowsH, numColsH = size(mH);

    if (convMode == CONV_MODE_FULL)
        numRowsFft  = numRowsI + numRowsH - 1;
        numColsFft  = numColsI + numColsH - 1;
        firstRowIdx = 1;
        firstColIdx = 1;
        lastRowIdx  = numRowsFft;
        lastColdIdx = numColsFft;
    elseif (convMode == CONV_MODE_SAME)
        numRowsFft  = numRowsI + numRowsH;
        numColsFft  = numColsI + numColsH;
        firstRowIdx = ceil((numRowsH + 1) / 2);
        firstColIdx = ceil((numColsH + 1) / 2);
        lastRowIdx  = firstRowIdx + numRowsI - 1;
        lastColdIdx = firstColIdx + numColsI - 1;
    elseif (convMode == CONV_MODE_VALID)
        numRowsFft  = numRowsI;
        numColsFft  = numColsI;
        firstRowIdx = numRowsH;
        firstColIdx = numColsH;
        lastRowIdx  = numRowsFft;
        lastColdIdx = numColsFft;
    end

    mT1[1:numRowsI, 1:numColsI] .= mI;
    mT2[1:numRowsH, 1:numColsH] .= mH;
    fft!(mT1);
    fft!(mT2);

    mT1 .*= mT2;
    ifft!(mT1);

    mO .= real.(@view mT1[firstRowIdx:lastRowIdx, firstRowIdx:lastRowIdx]);

end


## Parameters

# Data
imgUrl = "https://i.imgur.com/gx7rrPS.png"; #<! Lena image

# Kernel
kernelRadius = 2;

# Convolution
padMode     = PAD_MODE_REPLICATE;
convMode    = CONV_MODE_VALID;


## Generate / Load Data
mT = load(download(imgUrl));
tuSize = size(mT);

mI = ConvertJuliaImgArray(mT);
mI = mI ./ 255.0;

kernelLen = 2kernelRadius + 1;
mH = ones(kernelLen, kernelLen);
mH ./= sum(mH);

## Analysis

# Padding
mIPad = PadArray(mI, (kernelRadius, kernelRadius); padMode = padMode);

# Generating the Reference
mORef = Conv2D(mIPad, mH; convMode =  convMode);

# Using FFT
mODft = Conv2DFreqDom(mIPad, mH; convMode =  convMode);

## Display Results

figureIdx += 1;

oTr1 = heatmap(z = UInt8.(round.(255 * mI))[end:-1:1, :], showscale = false, colorscale = "Greys");
oLayout = Layout(title = "Input Image", width = tuSize[2] + 100, height = tuSize[1] + 100, hovermode = "closest", 
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));

hP = plot([oTr1], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr1 = heatmap(z = UInt8.(round.(255 * mORef))[end:-1:1, :], showscale = false, colorscale = "Greys");
oLayout = Layout(title = "Reference Image", width = tuSize[2] + 100, height = tuSize[1] + 100, hovermode = "closest", 
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));

hP = plot([oTr1], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTr1 = heatmap(z = UInt8.(round.(255 * mODft))[end:-1:1, :], showscale = false, colorscale = "Greys");
oLayout = Layout(title = "Using DFT", width = tuSize[2] + 100, height = tuSize[1] + 100, hovermode = "closest", 
                 margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));

hP = plot([oTr1], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
