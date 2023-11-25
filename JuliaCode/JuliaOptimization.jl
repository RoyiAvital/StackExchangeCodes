# StackExchange Code - Julia Optimization
# Set of functions for Optimization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.0.001     24/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added vanilla gradient descent (No acceleration).
#   *   Added allocating variations.
# - 1.0.000     09/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External


## Constants & Configuration

## Functions

function GradientDescent( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇ObjFun :: Function; ProjFun :: Function = identity ) where {T <: AbstractFloat, S <: Integer}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    for ii ∈ 1:numIter    
        vX .= ProjFun(vX .- (η .* ∇ObjFun(vX)));
    end

end

function GradientDescent!( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇vX :: AbstractVecOrMat{T}, ∇ObjFun! :: Function; ProjFun! :: Function = identity ) where {T <: AbstractFloat, S <: Integer}
    # This variation does not allocates memory.
    # Require from ∇ObjFun, ProjFun to be allocations free.

    for ii ∈ 1:numIter
        ∇ObjFun!(∇xX, vX);
    
        vX .= mX .- (η .* ∇vX);
        ProjFun!(vX);
    end

end

function GradientDescentAccelerated( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇ObjFun :: Function; ProjFun :: Function = identity ) where {T <: AbstractFloat, S <: Integer}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    vW = Array{T, length(size(vX))}(undef, size(vX));
    vZ = copy(vX);

    ∇vZ = Array{T, length(size(vX))}(undef, size(vX));

    for ii ∈ 1:numIter
        # FISTA (Nesterov) Accelerated
    
        ∇vZ = ∇ObjFun(vZ);
    
        vW .= vX; #<! Previous iteration
        vX .= vZ .- (η .* ∇vZ);
        vX .= ProjFun(vX);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW))
    end

end

function GradientDescentAccelerated!( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, vW :: AbstractVecOrMat{T}, vZ :: AbstractVecOrMat{T}, ∇vZ :: AbstractVecOrMat{T}, ∇ObjFun! :: Function; ProjFun! :: Function = identity ) where {T <: AbstractFloat, S <: Integer}
    # This variation does not allocates memory.
    # Require from ∇ObjFun, ProjFun to be allocations free.

    for ii ∈ 1:numIter
        # FISTA (Nesterov) Accelerated
    
        ∇ObjFun!(∇vZ, vZ);
    
        vW .= vX; #<! Previous iteration
        vX .= vZ .- (η .* ∇vZ);
        ProjFun!(vX);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW))
    end

end

