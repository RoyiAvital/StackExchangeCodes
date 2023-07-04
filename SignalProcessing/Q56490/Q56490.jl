# StackExchange Signal Processing Q56490
# https://dsp.stackexchange.com/questions/56490
# Implement Separable Bilateral Filter 
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  Restart REPL: Ctrl+J, CTRL+R (See https://stackoverflow.com/questions/74837851).
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     02/07/2023  Royi Avital
#   *   First release.

## Environment

# ENV["GRDISPLAY"] = "plot";

## Packages

# Internal
# using Printf;
# External
using ColorTypes;
using FileIO;
# import FreeType;
# using UnicodePlots;
using GR;

## Constants & Configuration

GR_CMP_GRAYSCALE = 2 #<! See https://github.com/jheinen/GR.jl/blob/master/examples/colormaps.jl

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions

function ConvertJuliaImgArray(mI :: Matrix{<: Color{T, N}}) where {T, N}
    # According to ColorTypes data always in order RGBA

    # println("RGB");
    
    numRows, numCols = size(mI);
    numChannels = N;
    dataType = T.types[1];

    mO = permutedims(reinterpret(reshape, dataType, mI), (2, 3, 1));

    if numChannels > 1
        mO = permutedims(reinterpret(reshape, dataType, mI), (2, 3, 1));
    else
        mO = reinterpret(reshape, dataType, mI);
    end

    return mO;

end

function ConvertJuliaImgArray(mI :: Matrix{<: Color{T, 1}}) where {T}
    # According to ColorTypes data always in order RGBA
    # Single Channel Image (numChannels = 1)
    
    # println("Gray");
    
    numRows, numCols = size(mI);
    dataType = T.types[1];

    mO = reinterpret(reshape, dataType, mI);

    return mO;

end

function BilateralFilter!( vO :: AbstractVector{T}, vI :: AbstractVector{T}, vB :: AbstractVector{T}, σₛ :: T, σₖ :: T, filterRadius :: U ) where{T <: AbstractFloat, U <: Unsigned}
    # Subscripts see https://stackoverflow.com/questions/36358017
    # Must use `AbstractVector` to support views
    
    numElements = length(vI);

    vB .= vI;

    for kk in 1:numElements
        leftRadius  = max(1, kk - filterRadius);
        rightRadius = min(numElements, kk + filterRadius);
        sumPx       = zero(T);
        sumW        = zero(T);
        for ii in leftRadius:rightRadius
            wⱼ     = exp( -0.5 * (((ii - kk) ^ 2) / (σₛ * σₛ)) -0.5 *  (((vI[ii] - vI[kk]) ^ 2) / (σₖ * σₖ)) )
            sumW  += wⱼ;
            sumPx += wⱼ * vI[ii];
        end
        vO[kk] = sumPx / sumW;
    end

end

function BilateralFilter!( mO :: Matrix{T}, mI :: Matrix{T}, σₛ :: T, σₖ :: T, filterRadius :: U ) where{T <: AbstractFloat, U <: Unsigned}
    # Subscripts see https://stackoverflow.com/questions/36358017
    
    numRows, numCols = size(mI);

    vB = Vector{T}(undef, max(numRows, numCols)); #<! Buffer for the in place operation

    for jj = 1:numCols
        @views BilateralFilter!(mO[:, jj], mI[:, jj], vB, σₛ, σₖ, filterRadius);
    end
    for ii = 1:numRows
        @views BilateralFilter!(mO[ii, :], mO[ii, :], vB, σₛ, σₖ, filterRadius);
    end

end


## Parameters

imgUrl = "https://i.imgur.com/gx7rrPS.png";

σₛ              = 3.0;
σₖ              = 0.15;
filterRadius    = ceil(UInt32, 3 * σₛ);

## Load / Generate Data

mT = load(download(imgUrl));
tSize = size(mT);

mI = ConvertJuliaImgArray(mT);
mI = mI ./ 255.0;


## Analysis

mO = Matrix{eltype(mI)}(undef, size(mI));
BilateralFilter!(mO, mI, σₛ, σₖ, filterRadius);


## Display Results


# Input Image
figure();
imshow(mI, colormap = GR_CMP_GRAYSCALE);

# Filtered Image
figure();
imshow(mO, colormap = GR_CMP_GRAYSCALE);


# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end
