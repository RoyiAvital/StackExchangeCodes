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

## Functions

function GradientDescentAccelerated!( mX :: AbstractVecOrMat{T}, numIter :: S, η :: T, mW :: AbstractVecOrMat{T}, mZ :: AbstractVecOrMat{T}, ∇mZ :: AbstractVecOrMat{T}, ∇ObjFun! :: Function, ProjFun! :: Function = identity ) where {T <: AbstractFloat, S <: Integer}

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

