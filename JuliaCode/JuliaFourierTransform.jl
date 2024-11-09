# StackExchange Code - Julia Fourier Transform
# Set of functions for Fourier Transform.
# References:
#   1.  
# Remarks:
#   1.  Optimize `@inbounds` and `@fastmath` for convolution.
# TODO:
# 	1.  Match `FftPad()` to MATLAB for `numSamples < numSamplesX`.
# Release Notes
# - 1.0.000     08/11/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;

# External
using FFTW;


## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

## Functions

function GenDftMatrix( numRows :: N, numCols :: N, :: Type{T} = Float64; normMat :: Bool = true ) where {N <: Integer, T <: AbstractFloat}

    # function GenDftMatrix( numRows :: N, numCols :: N, dataType :: DataType = Float64, normMat :: Bool = true ) where {N <: Integer}
    # if !(dataType <: AbstractFloat)
    #     throw(ArgumentError(lazy"The value of `dataType` must be a sub type of `AsbtractFloat`"));
    # end

    mF = Array{Complex{T}}(undef, numRows, numCols);
    for jj in 0:(numRows - 1), kk in 0:(numCols - 1)
        mF[jj + 1, kk + 1] = exp(-2im * π * jj * kk / numRows);
    end

    if normMat
        mF ./= sqrt(T(numCols));
    end

    return mF;
    
end

function FftPad( vX :: Vector{T}; numSamples :: N = length(vX) ) where {T <: AbstractFloat, N <: Integer}
    # Matches MATLAB's `fft(vX, n);`.

	numSamplesX = length(vX);
    numSamplesY = min(numSamplesX, numSamples);

	if (numSamples != numSamplesX)
        vXX = zeros(T, numSamples);
		vXX[1:numSamplesY] = vX[1:numSamplesY];
	else
		vXX = vX;
	end
	
    vF = fft(vXX);

	return vF;

end