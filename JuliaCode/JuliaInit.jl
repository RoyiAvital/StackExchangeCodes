# StackExchange Code - Julia Init
# Julia configuration and auxiliary functions.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.2.000     14/09/2023  Royi Avital
#   *   Added `ImgMat`.
# - 1.1.000     08/09/2023  Royi Avital
#   *   Added `VecOrView` and `MatOrView`.
#   *   Added `isJuliaInit` to suppress multiple initializations.
# - 1.0.000     09/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra; #<! For types

# External

# Types
if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
# Support views for Matrices / Vectors.
# See https://discourse.julialang.org/t/34932.
# Views are SubArrays: https://docs.julialang.org/en/v1/devdocs/subarrays
# Limit their type, dimension and type of parent array.
# One could use `Base.require_one_based_indexing` to ensure 1 based indexing.
VecOrView{T} = Union{Vector{T}, SubArray{T, 1, <: Array{T}}} where {T};
MatOrView{T} = Union{Matrix{T}, SubArray{T, 2, <: Array{T}}} where {T};

AdjOrTrans = LinearAlgebra.AdjOrTrans;

# MATLAB Like
VectorM{T} = Union{Vector{T}, SubArray{T, 1, <: Array{T}}, AdjOrTrans{T, <: Vector{T}}, AdjOrTrans{T, <: SubArray{T, 1, <: Array{T}}}} where {T};
MatrixM{T} = Union{Matrix{T}, SubArray{T, 2, <: Array{T}}, AdjOrTrans{T, <: Matrix{T}}, AdjOrTrans{T, <: SubArray{T, 2, <: Array{T}}}} where {T};

ImgMat{T} = Union{Matrix{T}, Array{T, 3}} where{T};
end

# vT = rand(4);
# vV = @view(vT[1:3]);
# typeof(vV') <: VectorM

# mT = rand(4, 4);
# mV = @view(mT[1:3, 1:3]);
# typeof(mV) <: MatOrView
# typeof(mV) <: MatrixM
# typeof(mT') <: MatrixM
# typeof(mV') <: MatrixM
# typeof(transpose(mT)) <: MatrixM
# typeof(transpose(mV)) <: MatrixM
# typeof(transpose(mV')) <: MatrixM
# typeof(transpose(Complex.(mV)')) <: MatrixM #<! Won't work!
# typeof(transpose(Complex.(mV)')') <: MatrixM #<! Won't work!


## Constants & Configuration

@enum PadMode begin
    PAD_MODE_CIRCULAR
    PAD_MODE_CONSTANT
    PAD_MODE_REFLECT
    PAD_MODE_REPLICATE
    PAD_MODE_SYMMETRIC
end

@enum BoundaryMode begin
    BND_MODE_CIRCULAR
    BND_MODE_REPLICATE
    BND_MODE_SYMMETRIC
    BND_MODE_ZEROS
end

@enum ConvMode begin
    CONV_MODE_FULL
    CONV_MODE_SAME
    CONV_MODE_VALID
end

@enum FilterMode begin
    FILTER_MODE_CONVOLUTION
    FILTER_MODE_CORRELATION
end

@enum OriginLoc begin
    BOTTOM_LEFT
    TOP_LEFT
end

@enum DiffMode begin
    DIFF_MODE_BACKWARD
    DIFF_MODE_CENTRAL
    DIFF_MODE_COMPLEX
    DIFF_MODE_FORWARD
end

@enum ColorConvMat begin
    RGB_TO_YCGCO
    YCGCO_TO_RGB
    RGB_TO_YPBPR_SD
    YPBPR_TO_RGB_SD
    RGB_TO_YPBPR_HD
    YPBPR_TO_RGB_HD
    RGB_TO_YUV
    YUV_TO_RGB
    RGB_TO_YIQ
    YIQ_TO_RGB
end

@enum ConnMode begin
    CONN_MODE_4
    CONN_MODE_8
end


# Display UIntx numbers as integers
Base.show(io :: IO, x :: T) where {T <: Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x);

## Auxiliary Functions

## Set Init State

# Check id defined or equals to `true` before running `JuliaInit.jl`
isJuliaInit = true;
