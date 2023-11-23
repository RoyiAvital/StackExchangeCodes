# StackExchange Code - Julia Image Processing
# Set of functions for Image Processing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.1.000     09/07/2023  Royi Avital
#   *   Added 2D Convolution.
# - 1.0.000     09/07/2023  Royi Avital
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

function PadArray( mA :: Matrix{T}, padRadius :: Tuple{N, N}, padMode :: PadMode; padValue :: T = zero(T) ) where {T <: Real, N <: Unsigned}
    # Works on Matrix
    # TODO: Verify!!!
    # TODO: Extend ot Array{T, 3}.
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

function Conv2D( mI :: Matrix{T}, mK :: Matrix{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: Real}
    
    if (convMode == CONV_MODE_FULL)
        mO = Matrix{T}(undef, size(mA) .+ size(mK) .- (1, 1));
    elseif (convMode == CONV_MODE_SAME) #<! TODO
        mO = Matrix{T}(undef, size(mA));
    elseif (convMode == CONV_MODE_VALID)
        mO = Matrix{T}(undef, size(mA) .- size(mK) .+ (1, 1));
    end

    Conv2D!(mO, vA, mK; convMode);
    return mO;

end

function Conv2D!( mO :: Matrix{T}, mI :: Matrix{T}, mK :: Matrix{T}; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat}

    if (convMode == CONV_MODE_FULL)
        _Conv2D!(vO, vA, vB);
    elseif (convMode == CONV_MODE_SAME) #<! TODO
        _Conv2DSame!(vO, vA, vB);
    elseif (convMode == CONV_MODE_VALID)
        _Conv2DValid!(vO, vA, vB);
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

function _Conv2DValidSK!( mO :: Matrix{T}, mI :: Matrix{T}, mK :: Matrix{T} ) where {T <: AbstractFloat}
    # Using StaticKernels.jl

    numRows, numCols = size(mK);
    radV, radH = (numRows, numCols) .÷ 2;
    mH = Kernel{(1:numRows, 1:numCols)}(@inline mW -> dot(Tuple(mW), Tuple(mK)))

    map!(mH, mO, mI);

end


