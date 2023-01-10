# StackExchange Signal Processing Q84826
# https://dsp.stackexchange.com/questions/84826
# What Measure to Compare the Color Depth (Distribution of Colors) of Images
# Julia
# References:
#   1.  
# Remarks:
#   1.  B
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     12/10/2022
#   *   First release.

## Packages

using StatsBase;
using Images;

using Plots;
using Printf;

## Constants

const ENTROPY_MODE_CHANNEL  = 1; #<! Averages per channel calculation
const ENTROPY_VECTOR        = 2; #<! Treats the RGB data as a single vector per pixel

## General Parameters

figureIdx = 0;
# gr();
plotlyjs(); #<! Seems to handle the data 
# inspectdr();

## Functions

# function CalcImgEntropy( mI :: Array{UInt8, 3}, entropyMode :: Integer ) :: T where{T <: AbstractFloat}
function CalcImgEntropy( mI :: Array{UInt8, 3}, entropyMode :: Integer ) :: AbstractFloat

    numRows     = size(mI, 1);
    numCols     = size(mI, 2);
    numChannels = size(mI, 3);

    if numChannels > 8
        error("The input image `mI` must have number of chanels which is 8 or less")
    end

    if(numChannels == 1)
        imgEntropy = CalcEntropy(mI);
    else
        if(entropyMode == ENTROPY_MODE_CHANNEL)
            imgEntropy = 0;
            for ii in 1:numChannels
                imgEntropy = imgEntropy + CalcEntropy(vec(mI[:, :, ii]));
            end
            imgEntropy = imgEntropy / numChannels;
        elseif(entropyMode == ENTROPY_VECTOR)
            if(numChannels > 4)
                mI = UInt64.(mI);
            else
                mI = UInt32.(mI);
            end
            mD = copy(mI[:, :, 1]);
            for ii in 2:numChannels
                mD .+= (mI[:, :, ii] .<< (8 * (ii - 1)));
            end
            imgEntropy = CalcEntropy(mD[:]);
        end
    end

    return imgEntropy;

end

function CalcEntropy( vI :: Vector{U} ) :: Float64 where{U <: Unsigned}
    
    dP = countmap(vI); #<! Dictionary
    # vU = collect(keys(dP));
    vP = convert.(Float64, (collect(values(dP))));

    vP .= vP ./ sum(vP);

    valEntropy = -sum(vP .* log2.(vP));

    return valEntropy;

end

function ConverImagePlanarForm( mI :: Matrix{<: RGB{T}} ) :: Array{T, 3} where{T <: AbstractFloat}
    # Converts from Julia Images Packed from to Planar form:
    # R1G1B1R2G2B2R3G3B3 -> R1R2R3G1G2G3B1B2B3

    return permutedims(reinterpret(reshape, T, mI), (2, 3, 1));

end

function ConverImagePlanarForm( mI :: Matrix{<: RGB{<: Normed{U}}} ) :: Array{U, 3} where{U <: Unsigned}
    # Converts from Julia Images Packed from to Planar form:
    # R1G1B1R2G2B2R3G3B3 -> R1R2R3G1G2G3B1B2B3

    return permutedims(reinterpret(reshape, U, mI), (2, 3, 1));

end

function ConverImagePlanarForm( mI :: Matrix{<: Gray{T}} ) :: Array{T, 2} where{T <: AbstractFloat}
    # Converts from Julia Images Packed from to Planar form:
    # R1G1B1R2G2B2R3G3B3 -> R1R2R3G1G2G3B1B2B3

    return reinterpret(T, mI);

end

# function ConverImagePlanarForm( mI :: Matrix{Gray{T}} ) :: Array{U, 2} where{T <: FixedPoint, U <: Unsigned}
#     # Converts image into Matrix

#     dataType = FixedPointNumbers.rawtype(eltype(eltype(mI)));
#     return reinterpret(dataType, mI);

# end


# function ConverImagePlanarForm( mI :: Matrix{Gray{T}} ) :: Array{U, 2} where{T <: FixedPoint, U <: Unsigned}
#     # Converts image into Matrix

#     dataType = FixedPointNumbers.rawtype(eltype(eltype(mI)));
#     return reinterpret(dataType, mI);

# end


## Parameters

img001Url = "https://i.stack.imgur.com/mYxDD.jpg";
img002Url = "https://i.stack.imgur.com/C95vE.jpg";


## Generate / Load Data

mI001 = load(download(img001Url));
mI002 = load(download(img002Url));

## Analysis

# Converting into planar form
mI001 = ConverImagePlanarForm(mI001);
mI002 = ConverImagePlanarForm(mI002);

img001Entropy1 = CalcImgEntropy(mI001, ENTROPY_MODE_CHANNEL);
img001Entropy2 = CalcImgEntropy(mI001, ENTROPY_VECTOR);
img002Entropy1 = CalcImgEntropy(mI002, ENTROPY_MODE_CHANNEL);
img002Entropy2 = CalcImgEntropy(mI002, ENTROPY_VECTOR);

println("The avergae per channel entropy of Img001 is: $img001Entropy1.");
println("The avergae per channel entropy of Img002 is: $img002Entropy1.");
println("The vectorized entropy of Img001 is: $img001Entropy2");
println("The vectorized entropy of Img002 is: $img002Entropy2");

## Display Results

# figureIdx += 1;
# fileName = @sprintf "Figure%04d.png" figureIdx;
# hP = plot(vA, abs.(vH), proj = :polar, label = "Antenna Spatial Gain", size = (1050, 700));
# display(hP);
# png(hP, fileName);