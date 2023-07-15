# StackExchange Code - Julia Image Processing
# Set of functions for Image Processing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     09/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal

# External
using ColorTypes;

## Constants & Configuration

include("./JuliaInit.jl");

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

function PadArray( mA :: Matrix{T}, padRadius :: Tuple{S, S}, padMode :: PadMode, padValue :: T = zero(T) ) where {T}
    # Works on Matrix
    # TODO: Verify!!!
    # TODO: Extend ot Array{T, 3}ץ
    # TODO: Create non allocating variant.

    numRows, numCols = size(mA);
    
    if (padMode == PadMode.CONSTANT)
        mB = Matrix{T}(undef, numRows + padRadius[1], numCols + padRadius[2]);
        mB .= padValue;
        mB[ee, cc] .= mA;
    end

    if (padMode == PadMode.REPLICATE)
        mB = Matrix{T}(undef, numRows + padRadius[1], numCols + padRadius[2]);
        for jj in 1:(numCols + padRadius[2])
            nn = clamp(jj - padRadius[2], 1, numCols);
            for ii in 1:(numRows + padRadius[1]) 
                mm = clamp(ii - padRadius[1], 1, numRows);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PadMode.SYMMETRIC)
        mB = Matrix{T}(undef, numRows + padRadius[1], numCols + padRadius[2]);
        for jj in 1:(numCols + padRadius[2])
            nn = ifelse(jj - padRadius[2] < 1, padRadius[2] - jj + 1, jj);
            nn = ifelse(jj - padRadius[2] > numCols, 2 * numCols - (jj - (numCols + padRadius[2]) - 1), jj);
            for ii in 1:(numRows + padRadius[1]) 
                mm = ifelse(ii - padRadius[1] < 1, padRadius[1] - ii + 1, ii);
                mm = ifelse(ii - padRadius[1] > numRows, 2 * numRows - (ii - (numRows + padRadius[1]) - 1), ii);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PadMode.REFLECT)
        mB = Matrix{T}(undef, numRows + padRadius[1], numCols + padRadius[2]);
        for jj in 1:(numCols + padRadius[2])
            nn = ifelse(jj - padRadius[2] < 1, padRadius[2] - jj + 2, jj);
            nn = ifelse(jj - padRadius[2] > numCols, 2 * numCols - (jj - (numCols + padRadius[2])), jj);
            for ii in 1:(numRows + padRadius[1]) 
                mm = ifelse(ii - padRadius[1] < 1, padRadius[1] - ii + 2, ii);
                mm = ifelse(ii - padRadius[1] > numRows, 2 * numRows - (ii - (numRows + padRadius[1])), ii);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PadMode.CIRCULAR)
        mB = Matrix{T}(undef, numRows + padRadius[1], numCols + padRadius[2]);
        for jj in 1:(numCols + padRadius[2])
            nn = ifelse(jj - padRadius[2] < 1, numCols + jj - padRadius[2], jj);
            nn = ifelse(jj - padRadius[2] > numCols, jj - numCols + padRadius[2], jj);
            for ii in 1:(numRows + padRadius[1]) 
                mm = ifelse(ii - padRadius[1] < 1, numRows + ii - padRadius[1], ii);
                mm = ifelse(ii - padRadius[1] > numRows, ii - numRows + padRadius[1], ii);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    return mB;

end