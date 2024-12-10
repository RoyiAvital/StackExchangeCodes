# StackExchange Code - Julia Optimization - Prox Operators
# Set of functions for Optimization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.0.000     09/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;

# External


## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

## Functions

function ProjSimplexBall!( vX :: AbstractVector{T}, vY :: AbstractVector{T}; ballRadius :: T = T(1.0), ε :: T = T(1e-7) ) where {T <: AbstractFloat}
    #TODO: Make zero allocations
    
    numElements = length(vY);

    if (length(vX) != numElements)
        throw(DimensionMismatch(lazy"The length of `vX` `vY` must match"));
    end
    
    copy!(vX, vY);

    if ((abs(sum(vY) - ballRadius) < ε) && all(vY .>= zero(T)))
        # The input is already within the Simplex.        
        return vX;
    end

    sort!(vX); #<! TODO: Make inplace

    # Breakpoints of the piecewise function happens at xᵢ - μ = 0α → Search for points xᵢ - 0 
    vμ         = vcat(vX[1] - ballRadius, vX, vX[numElements] + ballRadius);
    hObjFun(μ) = sum(max.(vY .- μ, zero(T))) - ballRadius;

    vObjVal = zeros(numElements + 2);
    for ii = 1:(numElements + 2)
        vObjVal[ii] = hObjFun(vμ[ii]);
    end

    if (any(vObjVal .== zero(T)))
        μ = vμ(vObjVal .== zero(T));
    else
        # Working on when an Affine Function have the value zero
        valX1Idx = findlast(>(zero(T)), vObjVal);
        valX2Idx = findfirst(<(zero(T)), vObjVal);
    
        valX1 = vμ[valX1Idx];
        valX2 = vμ[valX2Idx];
        valY1 = vObjVal[valX1Idx];
        valY2 = vObjVal[valX2Idx];
    
        # Linear Function, Intersection with Zero
        paramA = (valY2 - valY1) / (valX2 - valX1);
        paramB = valY1 - (paramA * valX1);
        μ      = -paramB / paramA;
    end

    @. vX = max(vY - μ, zero(T));

    return vX;

end

function ProjSimplexBall( vY :: AbstractVector{T}; ballRadius :: T = T(1.0), ε :: T = T(1e-7) ) where {T <: AbstractFloat}
    
    numElements = length(vY);
    vX = zeros(T, numElements);

    return ProjSimplexBall!(vX, vY; ballRadius = ballRadius, ε = ε);

end

function ProjectL1Ball!(vX :: Vector{T}, vY :: Vector{T}, ballRadius :: T) where {T <: AbstractFloat}

    numElements = length(vY);
    
    if(sum(abs, vY) <= ballRadius)
        # The input is already within the L1 Ball.
        vX .= vY;
        return vX;
    end

    vX .= abs.(vY);
    sort!(vX);

    λ          = T(-1.0);
    xPrev      = zero(T);
    objValPrev = ProjectL1BallObj(vX, xPrev, ballRadius);

    for ii in 1:numElements
        objVal = ProjectL1BallObj(vX, vX[ii], ballRadius);
        if (objVal == zero(T))
            λ = vZ[ii];
            break;
        end

        if (objVal < zero(T))
            paramA = (objVal - objValPrev) / (vX[ii] - xPrev);
            paramB = objValPrev - (paramA * xPrev);
            λ = -paramB / paramA;
            break;
        end
        xPrev      = vX[ii];
        objValPrev = objVal;
    end

    if (λ < zero(T))
        # The last value isn't large enough
        objVal = ProjectL1BallObj(vX, vX[numElements] + ballRadius, ballRadius);
        paramA = (objVal - objValPrev) / (vX[numElements] + ballRadius - xPrev);
        paramB = objValPrev - (paramA * xPrev);
        λ = -paramB / paramA;
    end
    
    @. vX = sign(vY) * max(abs(vY) - λ, zero(T));

    return vX;

end

function ProjectL1BallObj( vZ :: Vector{T}, λ :: T, ballRadius :: T ) where {T <: AbstractFloat}

    objVal = zero(T);
    for ii in 1:length(vY)
        objVal += max(vZ[ii] - λ, zero(T));
    end

    objVal -= ballRadius;

    return objVal;
    
end

function ProjectL1Ball( vY :: AbstractVector{T}; ballRadius :: T = T(1.0), ε :: T = T(1e-7) ) where {T <: AbstractFloat}
    
    numElements = length(vY);
    vX = zeros(T, numElements);

    return ProjectL1Ball!(vX, vY; ballRadius = ballRadius, ε = ε);

end
