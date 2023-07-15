# StackExchange Code - Julia Optimization
# Set of functions for Optimization.
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


## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions

function GradientDescentAccelerated!( mX :: Array{T, D}, numIter :: S, η :: T, mW :: Array{T, D}, mZ :: Array{T, D}, ∇mZ :: Array{T, D}, ∇ObjFun! :: Function, ProjFun! :: Function = identity ) where {T <: AbstractFloat, D <: 2, S <: Integer}

    for ii ∈ 1:numIter
        # FISTA (Nesterov) Accelerated
    
        ∇ObjFun!(∇mZ, mZ);
    
        mW .= mX; #<! Previous iteration
        mX .= mZ .- (η .* ∇mZ);
        ProjFun!(mX);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        mZ .= mX .+ (fistaStepSize .* (mX .- mW))
    end

end

