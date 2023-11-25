# StackExchange Code - Julia Image Processing
# Set of functions for Image Processing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  Add convolution in frequency domain as in DSP `Q90036`.
#       It should include an auxiliary function: `GenWorkSpace()` for teh buffers.  
#       It should also be optimized for `rfft()`.
# Release Notes
# - 1.1.000     23/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added 2D Convolution.
# - 1.0.000     09/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External
using ColorTypes;
using LoopVectorization;
using StaticKernels;

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

function PadArray( mA :: Matrix{T}, tuPadRadius :: Tuple{N, N}; padMode :: PadMode, padValue :: T = zero(T) ) where {T <: Real, N <: Signed}
    # Works on Matrix
    # TODO: Support padding larger then the input.
    # TODO: Extend ot Array{T, 3}.
    # TODO: Create non allocating variant.

    numRowsA, numColsA = size(mA);
    numRowsB, numColsB = (numRowsA, numColsA) .+ (2 .* tuPadRadius);
    mB = Matrix{T}(undef, numRowsB, numColsB);
    
    if (padMode == PAD_MODE_CONSTANT)
        mB .= padValue;
        mB[(tuPadRadius[1] + 1):(numRowsB - tuPadRadius[1]), (tuPadRadius[2] + 1):(numColsB - tuPadRadius[2])] .= mA;
    end

    if (padMode == PAD_MODE_REPLICATE)
        for jj in 1:numColsB
            nn = clamp(jj - tuPadRadius[2], 1, numColsA);
            for ii in 1:numRowsB
                mm = clamp(ii - tuPadRadius[1], 1, numRowsA);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PAD_MODE_SYMMETRIC)
        for jj in 1:numColsB
            nn = ifelse(jj - tuPadRadius[2] < 1, tuPadRadius[2] - jj + 1, jj - tuPadRadius[2]);
            nn = ifelse(jj - tuPadRadius[2] > numColsA, numColsA - (jj - (numColsA + tuPadRadius[2]) - 1), nn);
            for ii in 1:numRowsB
                mm = ifelse(ii - tuPadRadius[1] < 1, tuPadRadius[1] - ii + 1, ii - tuPadRadius[1]);
                mm = ifelse(ii - tuPadRadius[1] > numRowsA, numRowsA - (ii - (numRowsA + tuPadRadius[1]) - 1), mm);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PAD_MODE_REFLECT) #<! Mirror
        for jj in 1:numColsB
            nn = ifelse(jj - tuPadRadius[2] < 1, tuPadRadius[2] - jj + 2, jj - tuPadRadius[2]);
            nn = ifelse(jj - tuPadRadius[2] > numColsA, numColsA - (jj - (numColsA + tuPadRadius[2])), nn);
            for ii in 1:numRowsB
                mm = ifelse(ii - tuPadRadius[1] < 1, tuPadRadius[1] - ii + 2, ii - tuPadRadius[1]);
                mm = ifelse(ii - tuPadRadius[1] > numRowsA, numRowsA - (ii - (numRowsA + tuPadRadius[1])), mm);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    if (padMode == PAD_MODE_CIRCULAR)
        for jj in 1:numColsB
            nn = ifelse(jj - tuPadRadius[2] < 1, numColsA + jj - tuPadRadius[2], jj - tuPadRadius[2]);
            nn = ifelse(jj - tuPadRadius[2] > numColsA, jj - numColsA - tuPadRadius[2], nn);
            for ii in 1:numRowsB
                mm = ifelse(ii - tuPadRadius[1] < 1, numRowsA + ii - tuPadRadius[1], ii - tuPadRadius[1]);
                mm = ifelse(ii - tuPadRadius[1] > numRowsA, ii - numRowsA - tuPadRadius[1], mm);
                mB[ii, jj] = mA[mm, nn];
            end
        end
    end

    return mB;

end

function Conv2D( mI :: Matrix{T}, mK :: Matrix{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: Real}
    
    if (convMode == CONV_MODE_FULL)
        mO = Matrix{T}(undef, size(mI) .+ size(mK) .- (1, 1));
    elseif (convMode == CONV_MODE_SAME) #<! TODO
        mO = Matrix{T}(undef, size(mI));
    elseif (convMode == CONV_MODE_VALID)
        mO = Matrix{T}(undef, size(mI) .- size(mK) .+ (1, 1));
    end

    Conv2D!(mO, mI, mK; convMode = convMode);
    return mO;

end

function Conv2D!( mO :: Matrix{T}, mI :: Matrix{T}, mK :: Matrix{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat}

    if (convMode == CONV_MODE_FULL)
        _Conv2D!(mO, mI, mK);
    elseif (convMode == CONV_MODE_SAME) #<! TODO
        _Conv2DSame!(mO, mI, mK);
    elseif (convMode == CONV_MODE_VALID)
        _Conv2DValid!(mO, mI, mK);
    end
    
end

function _Conv2D!( mO :: Matrix{T}, mI :: Matrix{T}, mK :: Matrix{T} ) where {T <: AbstractFloat}

    numRowsI, numColsI = size(mI);
    numRowsK, numColsK = size(mK);

    for jj ∈ 1:(numColsK - 1), ii ∈ 1:(numRowsK - 1) #<! Top Left
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj >= nn) && (ii >= mm);
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ 1:(numColsK - 1), ii ∈ numRowsK:(numRowsI - 1) #<! Middle Left
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj >= nn);
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ 1:(numColsK - 1), ii ∈ numRowsI:(numRowsI + numRowsK - 1) #<! Bottom Left
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj >= nn) && (ii < numRowsI + mm);;
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ numColsK:(numColsI - 1), ii ∈ 1:(numRowsK - 1) #<! Top Middle
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (ii >= mm);
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ numColsK:(numColsI - 1)
        @turbo for ii ∈ numRowsK:(numRowsI - 1) #<! Middle Middle
            sumVal = zero(T);
            for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
                @inbounds sumVal += mK[mm, nn] * mI[ii - mm + 1, jj - nn + 1];
            end
            mO[ii, jj] = sumVal;
        end        
    end

    for jj ∈ numColsK:(numColsI - 1), ii ∈ numRowsI:(numRowsI + numRowsK - 1) #<! Bottom Middle
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (ii < numRowsI + mm);;
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ numColsI:(numColsI + numColsK - 1), ii ∈ 1:(numRowsK - 1) #<! Top Right
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj < numColsI + nn) && (ii >= mm);
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ numColsI:(numColsI + numColsK - 1), ii ∈ numRowsK:(numRowsI - 1) #<! Middle Right
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj < numColsI + nn);
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

    for jj ∈ numColsI:(numColsI + numColsK - 1), ii ∈ numRowsI:(numRowsI + numRowsK - 1) #<! Bottom Right
        sumVal = zero(T);
        for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
            ib0 = (jj < numColsI + nn) && (ii < numRowsI + mm);;
            @inbounds oa = ib0 ? mI[ii - mm + 1, jj - nn + 1] : zero(T);
            @inbounds sumVal += mK[mm, nn] * oa;
        end
        mO[ii, jj] = sumVal;
    end

end

# TODO: Add the `same` variant.

function _Conv2DValid!( mO :: Matrix{T}, mI :: Matrix{T}, mK :: Matrix{T} ) where {T <: AbstractFloat}

    numRowsI, numColsI = size(mI);
    numRowsK, numColsK = size(mK);

    for jj ∈ 1:(numColsI - numColsK + 1)
        @turbo for ii in 1:(numRowsI - numRowsK + 1)
            sumVal = zero(T);
            for nn ∈ 1:numColsK, mm ∈ 1:numRowsK
                @inbounds sumVal += mK[mm, nn] * mI[ii - mm + numRowsK, jj - nn + numColsK];
            end
            mO[ii, jj] = sumVal;
        end
    end

end


