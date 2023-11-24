# StackExchange Signal Processing Q81420
# https://dsp.stackexchange.com/questions/81420
# Apply 2D Convolution on an Image Using Tiles
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
# - 1.0.000     07/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal

# External
using CairoMakie;
using ColorTypes;
using FileIO;
using StaticKernels;

## Constants & Configuration

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

function PadImage( mI :: Matrix{T}, tPad :: Tuple{<: Integer, <: Integer} ) where {T}
    # Nearest Neighbor (Replicate)
    
    numRows, numCols = size(mI);
    
    vI = clamp.((1 - tPad[1]):(numRows + tPad[1]), 1, numRows);
    vJ = clamp.((1 - tPad[2]):(numCols + tPad[2]), 1, numCols);
    
    return mI[vI, vJ];
end


## Parameters

# Data
imgUrl = "https://i.imgur.com/gx7rrPS.png";

# Tiles
tTileSize = (64, 64);

# Kernel
kernelRadius = 5;


## Load / Generate Data

mT = load(download(imgUrl));
tSize = size(mT);

mI = ConvertJuliaImgArray(mT);
mI = mI ./ 255.0;


## Analysis
kernelLen = (2 * kernelRadius) + 1;
kernelNumPx = Float64(kernelLen * kernelLen);
mK      = Kernel{(-kernelRadius:kernelRadius, -kernelRadius:kernelRadius)}(@inline mW -> (sum(mW) / kernelNumPx));
mIPad   = PadImage(mI, (kernelRadius, kernelRadius));

mORef = map(mK, mIPad);
mORef = Float64.(mORef); #<! See https://github.com/stev47/StaticKernels.jl/issues/11

mO = zeros(size(mI));

# Working by the inner indices
for firstRowIdx in 1:tTileSize[1]:tSize[1], firstColIdx in 1:tTileSize[2]:tSize[2]
    lastRowIdx  = firstRowIdx + tTileSize[1] - 1;
    lastColIdx  = firstColIdx + tTileSize[2] - 1;
    
    # The padded image is basically shifted by `kernelRadius`
    firstRowIdxTile = firstRowIdx;
    lastRowIdxTile  = firstRowIdxTile + tTileSize[1] + (2 * kernelRadius) - 1;
    firstColIdxTile = firstColIdx;
    lastColIdxTile  = firstColIdxTile + tTileSize[2] + (2 * kernelRadius) - 1;
    @views map!(mK, mO[firstRowIdx:lastRowIdx, firstColIdx:lastColIdx], mIPad[firstRowIdxTile:lastRowIdxTile, firstColIdxTile:lastColIdxTile]);
end

## Display Results

hF = Figure(resolution = tSize);
image!(hF.scene, 0..tSize[1], 0..tSize[2], rotr90(mI));
display(hF);
sleep(0.1); #<! Allow enough time to write to HD and display the image

hF = Figure(resolution = tSize);
image!(hF.scene, 0..tSize[1], 0..tSize[2], rotr90(mORef));
display(hF);
sleep(0.1); #<! Allow enough time to write to HD and display the image

hF = Figure(resolution = tSize);
image!(hF.scene, 0..tSize[1], 0..tSize[2], rotr90(mO));
display(hF);
sleep(0.1); #<! Allow enough time to write to HD and display the image
