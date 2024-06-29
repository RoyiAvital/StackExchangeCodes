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
# - 1.2.000     29/06/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added functions to generate Convolution Matrix (Sparse).
#   *   Added a function to calculate the image Laplace Operator.
# - 1.1.000     23/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added 2D Convolution.
# - 1.0.000     09/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using SparseArrays;

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
    # Full Convolution

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


function GenConvMtx( vK :: AbstractVector{T}, numElements :: N; convMode :: ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat, N <: Integer}
    
    kernelLength = length(vK);

    if (convMode == CONV_MODE_FULL)
        rowIdxFirst = 1;
        rowIdxLast  = numElements + kernelLength - 1;
        outputSize  = numElements + kernelLength - 1;
    elseif (convMode == CONV_MODE_SAME)
        rowIdxFirst = 1 + floor(kernelLength / 2);
        rowIdxLast  = rowIdxFirst + numElements - 1;
        outputSize  = numElements;
    elseif (convMode == CONV_MODE_VALID)
        rowIdxFirst = kernelLength;
        rowIdxLast  = (numElements + kernelLength - 1) - kernelLength + 1;
        outputSize  = numElements - kernelLength + 1;
    end

    mtxIdx = 0;
    vI = ones(N, numElements * kernelLength);
    vJ = ones(N, numElements * kernelLength);
    vV = zeros(T, numElements * kernelLength);

    for jj ∈ 1:numElements
        for ii ∈ 1:kernelLength
            if((ii + jj - 1 >= rowIdxFirst) && (ii + jj - 1 <= rowIdxLast))
                # Valid output matrix row index
                mtxIdx += 1;
                vI[mtxIdx] = ii + jj - rowIdxFirst;
                vJ[mtxIdx] = jj;
                vV[mtxIdx] = vK[ii];
            end
        end
    end

    mK = sparse(vI, vJ, vV, outputSize, numElements);

    return mK;

end


function GenConvMtx( mH :: AbstractMatrix{T}, numRows :: N, numCols :: N; convMode = ConvMode = CONV_MODE_FULL ) where {T <: AbstractFloat, N <: Integer}
    
    numColsKernel   = size(mH, 2);
    numBlockMtx     = numColsKernel;

    lBlockMtx = Vector{SparseMatrixCSC{T, N}}(undef, numBlockMtx);

    for ii ∈ 1:numBlockMtx
        lBlockMtx[ii] = GenConvMtx(mH[:, ii], numRows; convMode = convMode);
    end

    if (convMode == CONV_MODE_FULL)
        # For convolution shape - 'full' the Doubly Block Toeplitz Matrix
        # has the first column as its main diagonal.
        diagIdx     = 0;
        numRowsKron = numCols + numColsKernel - 1;
    elseif (convMode == CONV_MODE_SAME)
        # For convolution shape - 'same' the Doubly Block Toeplitz Matrix
        # has the first column shifted by the kernel horizontal radius.
        diagIdx     = floor(numColsKernel / 2);
        numRowsKron = numCols;
    elseif (convMode == CONV_MODE_VALID)
        # For convolution shape - 'valid' the Doubly Block Toeplitz Matrix
        # has the first column shifted by the kernel horizontal length.
        diagIdx     = numColsKernel - 1;
        numRowsKron = numCols - numColsKernel + 1;
    end

    vI = ones(N, min(numRowsKron, numCols));
    mK = kron(spdiagm(numRowsKron, numCols, diagIdx => vI), lBlockMtx[1]);

    for ii ∈ 2:numBlockMtx
        diagIdx = diagIdx - 1;
        mK = mK + kron(spdiagm(numRowsKron, numCols, diagIdx => vI), lBlockMtx[ii]);
    end

    return mK;

end


function CalcImageLaplacian!( mO :: Matrix{T}, mI :: Matrix{T}, mB :: Matrix{T}, mV1 :: Matrix{T}, mV2 :: Matrix{T}, mKₕ :: Matrix{T}, mKᵥ :: Matrix{T} ) where {T <: AbstractFloat}

    _Conv2DValid!(mV1, mI, mKₕ);
    _Conv2D!(mO, mV1, rot180(mKₕ)); #<! TODO: Remove allocation

    _Conv2DValid!(mV2, mI, mKᵥ);
    _Conv2D!(mB, mV2, rot180(mKᵥ)); #<! TODO: Remove allocation

    mO .+= mB;

    return mO;

end

function CalcImageLaplacian( mI :: Matrix{T}; mKₕ :: Matrix{T} = [one(T) -one(T)], mKᵥ :: Matrix{T} = [[one(T), -one(T)];;] ) where {T <: AbstractFloat}

    mO = similar(mI);
    mB = similar(mI); #<! Buffer

    numRows = size(mI, 1);
    numCols = size(mI, 2);

    numRowsKₕ = size(mKₕ, 1);
    numColsKₕ = size(mKₕ, 2);
    numRowsmKᵥ = size(mKᵥ, 1);
    numColsmKᵥ = size(mKᵥ, 2);

    mV1 = Matrix{T}(undef, numRows - numRowsKₕ + 1, numCols - numColsKₕ + 1);
    mV2 = Matrix{T}(undef, numRows - numRowsmKᵥ + 1, numCols - numColsmKᵥ + 1);

    mO = CalcImageLaplacian!(mO, mI, mB, mV1, mV2, mKₕ, mKᵥ);

    return mO;

end


