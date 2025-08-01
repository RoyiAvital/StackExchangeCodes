# StackExchange Code - Julia Optimization
# Set of functions for Optimization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.0.007     20/07/2025  Royi Avital RoyiAvital@yahoo.com
#   *   Added `FindZeroBinarySearch()`.
#   *   Added `LsqBox()`.
# - 1.0.006     04/11/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Fixed issued with `GradientDescentBackTracking()`.
#   *   Made explicit package usage.
# - 1.0.005     08/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Verifying the initialization happens only once.
# - 1.0.004     28/06/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Updated `ADMM()` and `ADMM!()`.
#   *   Added a function to project onto intersection of convex sets.
#   *   Added Accelerated Proximal Gradient Descent.
# - 1.0.003     21/01/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `IRLS()` and `IRLS!()` for || A x - b ||_p.
# - 1.0.002     27/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added `ADMM!()` for classic ADMM form.
# - 1.0.001     24/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   Added vanilla gradient descent (No acceleration).
#   *   Added allocating variations.
# - 1.0.000     09/07/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;

# External
using FastLapackInterface;


## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

include("./JuliaProxOperators.jl");

## Functions

function CalcFunGrad!( vG :: AbstractVecOrMat{T}, vP :: AbstractVecOrMat{T}, vX :: AbstractVecOrMat{T}, hFun :: Function; diffMode :: DiffMode = DIFF_MODE_CENTRAL, ε :: T = T(1e-6) ) where {T <: AbstractFloat}
    # Non allocating assuming `hFun()` is non allocating.
    # vP is assumed to be zeros.

    numElements = length(vX);
    funValRef   = hFun(vX);

    # It seems that Julia can not define local functions in `if` block
    # See https://github.com/JuliaLang/julia/issues/15602, https://discourse.julialang.org/t/13815
    # if (diffMode == DIFF_MODE_BACKWARD)
    #     DiffFun(vP) = (funValRef - hFun(vX - vP)) / ε;
    # elseif (diffMode == DIFF_MODE_CENTRAL) 
    #     DiffFun(vP) = (hFun(vX + vP) - hFun(vX - vP)) / (2ε);
    # elseif (diffMode == DIFF_MODE_COMPLEX) 
    #     DiffFun(vP) = imag(hFun(vX + (1im * vP))) / ε;
    # elseif (diffMode == DIFF_MODE_FORWARD) 
    #     DiffFun(vP) = (hFun(vX + vP) - funValRef) / ε;
    # end

    # Could be solved using anonymous functions.
    # One could use hDiffFun = vP :: AbstractVecOrMat{T} -> (funValRef - hFun(vX - vP)) / ε;.
    # Yet, performance wise, it won't do anything and `vP` is already guaranteed to be `AbstractVecOrMat{T}`.
    if (diffMode == DIFF_MODE_BACKWARD)
        DiffFun = vP -> (funValRef - hFun(vX - vP)) / ε;
    elseif (diffMode == DIFF_MODE_CENTRAL) 
        DiffFun = vP -> (hFun(vX + vP) - hFun(vX - vP)) / (2ε);
    elseif (diffMode == DIFF_MODE_COMPLEX) 
        DiffFun = vP -> imag(hFun(vX + (1im * vP))) / ε;
    elseif (diffMode == DIFF_MODE_FORWARD) 
        DiffFun = vP -> (hFun(vX + vP) - funValRef) / ε;
    end

    # Using actual function definition (Equivalent to the anonymous above).
    # Functions are processed as objects and then assigned.
    # if (diffMode == DIFF_MODE_BACKWARD)
    #     DiffFun = function( vP :: AbstractVecOrMat{T} ) 
    #         return (funValRef - hFun(vX - vP)) / ε;
    #     end
    # elseif (diffMode == DIFF_MODE_CENTRAL) 
    #     DiffFun = function( vP :: AbstractVecOrMat{T} )
    #         return (hFun(vX + vP) - hFun(vX - vP)) / (2ε);
    #     end
    # elseif (diffMode == DIFF_MODE_COMPLEX) 
    #     DiffFun = function( vP :: AbstractVecOrMat{T} )
    #         return imag(hFun(vX + (1im * vP))) / ε;
    #     end
    # elseif (diffMode == DIFF_MODE_FORWARD) 
    #     DiffFun = function( vP :: AbstractVecOrMat{T} )
    #         return (hFun(vX + vP) - funValRef) / ε;
    #     end
    # end

    for ii ∈ 1:numElements
        vP[ii] = ε;
        vG[ii] = DiffFun(vP);
        vP[ii] = zero(T);
    end

end

function CalcFunGrad( vX :: AbstractVecOrMat{T}, hFun :: Function; diffMode :: DiffMode = DIFF_MODE_CENTRAL, ε :: T = T(1e-6) ) where {T <: AbstractFloat}

    vP = zeros(T, size(vX));
    vG = zeros(T, size(vX));

    CalcFunGrad!(vG, vP, vX, hFun; diffMode = diffMode, ε = ε);

    return vG;

end


function GradientDescent( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇ObjFun :: Function; ProjFun :: Function = identity ) where {T <: AbstractFloat, S <: Integer}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    for ii ∈ 1:numIter    
        vX .= ProjFun(vX .- (η .* ∇ObjFun(vX)));
    end

    return vX;

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

function GradientDescentBackTracking!( vX :: AbstractVecOrMat{T}, vG :: AbstractVecOrMat{T}, vZ :: AbstractVecOrMat{T}, numIter :: S, η :: T, ObjFun :: Function, ∇ObjFun :: Function; α :: T = T(0.5), β :: T = T(1e-10) ) where {T <: AbstractFloat, S <: Integer}
    # Does not support projection (See `GradientDescentBackTrackingProj!()`, which adds a lot of overhead)
    # Non allocating assuming function are not allocating.

    for ii ∈ 1:numIter
        vG .= ∇ObjFun(vX);
        objFunVal = ObjFun(vX);
        vZ .= vX .- η .* vG;
        while ((ObjFun(vZ) > objFunVal) && (η > β))
            η *= α;
            vZ .= vX .- η .* vG;
        end
        vG .*= η;
        η   /= α;
        vX .-= vG;
    
    end

end

# function GradientDescentBackTrackingProj!( vX :: AbstractVecOrMat{T}, vG :: AbstractVecOrMat{T}, vZ :: AbstractVecOrMat{T}, numIter :: S, η :: T, ObjFun :: Function, ∇ObjFun :: Function; ProjFun :: Function = identity, α :: T = T(0.5), β :: T = T(1e-10) ) where {T <: AbstractFloat, S <: Integer}
#     # Non allocating assuming function are not allocating.

#     # Based on https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/08-prox-grad-scribed.pdf Page 4
#     hGradMap( vX :: Vector{T}, vG :: Vector{T}, α :: T ) = (vX .- ProjFun(vX .- α .* vG)) ./ α;

#     for ii ∈ 1:numIter
#         vG .= ∇ObjFun(vX);
#         objFunVal = ObjFun(vX);
#         vZ .= vX .- η .* hGradMap(vX, vG, η);
#         while ((ObjFun(vZ) > objFunVal - η * dot(vG, hGradMap(vX, vG, η)) + T(0.5) * η * sum(abs2, hGradMap(vX, vG, η))) && (η > β))
#             η *= α;
#             vZ .= vX .- η .* hGradMap(vX, vG, η);
#         end
#         # vG .*= η;
#         η   /= α;
#         # vX .-= vG;
    
#         vX .-= η .* hGradMap(vX, vG, η);
#     end

# end

function GradientDescentBackTracking( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ObjFun :: Function, ∇ObjFun :: Function; α :: T = T(0.5), β :: T = T(1e-10) ) where {T <: AbstractFloat, S <: Integer}
    # Mutates vX
    # Does not support projection (See `GradientDescentBackTrackingProj!()` which requires a lot of overhead).
    # Projection should have its own variant (Use dispatch, or different name).

    vG = copy(vX);
    vZ = copy(vX);

    GradientDescentBackTracking!(vX, vG, vZ, numIter, η, ObjFun, ∇ObjFun; α = α, β = β);

    return vX;

end


function GradientDescentAccelerated( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇ObjFun :: Function; ProjFun :: Function = identity ) where {T <: AbstractFloat, S <: Integer} #, F <: Function, G <: Function}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    vW = Array{T, ndims(vX)}(undef, size(vX));
    vZ = copy(vX);

    ∇vZ = Array{T, ndims(vX)}(undef, size(vX));

    for ii ∈ 1:numIter
        # FISTA (Nesterov) Accelerated
    
        ∇vZ = ∇ObjFun(vZ);
    
        vW .= vX; #<! Previous iteration
        vX .= vZ .- (η .* ∇vZ);
        vX .= ProjFun(vX);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW));
    end

    return vX;

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
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW));
    end

end

function ADMM!(vX :: AbstractVector{T}, vZ :: AbstractVector{T}, vU :: AbstractVector{T}, mA :: AbstractMatrix{T}, hProxF :: Function, hProxG :: Function, numIterations :: N; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat, N <: Integer}
    # Solves f(x) + λ g(Ax)
    # Where z = Ax, and g(z) has a well defined Prox.
    # ADMM for the case Ax + z = 0
    # ProxF(y) = \arg \minₓ 0.5ρ * || A x - y ||_2^2 + f(x)
    # ProxG(y) = \arg \minₓ 0.5ρ * || x - y ||_2^2 + λ g(x)
    # Initialization by mX[:, 1]
    # Supports in place ProxG
    
    for ii ∈ 1:numIterations
        vZ .-= vU;
        vX .= hProxF(vZ, ρ);
        mul!(vZ, mA, vX);
        vZ .+= vU;
        vZ .= hProxG(vZ, λ / ρ);
        # vX .= hProxF(vZ - vU, ρ);
        # vZ .= hProxG(mA * vX + vU, λ / ρ);
        vU .= mul!(vU, mA, vX, one(T), one(T)) .- vZ;
    end

    return vX;

end

function ADMM(vX :: AbstractVector{T}, mA :: AbstractMatrix{T}, hProxF :: Function, hProxG :: Function, numIterations :: N; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat, N <: Integer}
    # Solves f(x) + λ g(Ax)
    # Where z = Ax, and g(z) has a well defined Prox.
    # ADMM for the case Ax + z = 0
    # ProxF(y) = \arg \minₓ 0.5ρ * || A x - y ||_2^2 + f(x)
    # ProxG(y) = \arg \minₓ 0.5ρ * || x - y ||_2^2 + λ g(x)
    # Initialization by mX[:, 1]
    # Supports in place ProxG
    # https://nikopj.github.io/notes/admm_scaled
    # TODO: Add support for uniform scaling for mA.

    numRows = size(mA, 1);

    vZ = zeros(T, numRows);
    vU = zeros(T, numRows);

    vX = ADMM!(vX, vZ, vU, mA, hProxF, hProxG, numIterations; ρ = ρ, λ = λ);
    
    return vX;

end

function ProximalGradientDescent!( vX :: AbstractVector{T}, vG :: AbstractVector{T}, ∇F :: Function, ProxG :: Function, η :: T, numIterations :: S; λ :: T = one(T) ) where {T <: AbstractFloat, S <: Integer}
    # Solves f(x) + λ g(x)
    # ∇F(y) = ∇f(y)
    # ProxG(y) = \arg \minₓ 0.5 * || x - y ||_2^2 + λ g(x)
    # Supports in place ProxG

    λ *= η;

    for ii ∈ 1:numIterations
        vG = ∇F(vX);
        vX .-= η .* vG; 
        vX .= ProxG(vX, λ);
    end

    return vX;

end

function ProximalGradientDescent( vX :: AbstractVector{T}, ∇F :: Function, ProxFun :: Function, η :: T, numIterations :: S; λ :: T = one(T) ) where {T <: AbstractFloat, S <: Integer}

    vG = similar(vX);
    vX = ProximalGradientDescent!(vX, vG, ∇F, ProxFun, η, numIterations; λ = λ);

    return vX;

end

function ProximalGradientDescentAcc!( vX :: AbstractVector{T}, vG :: AbstractVector{T}, vZ :: AbstractVector{T}, vW :: AbstractVector{T}, ∇F :: Function, ProxG :: Function, η :: T, numIterations :: S; λ :: T = one(T) ) where {T <: AbstractFloat, S <: Integer}
    # Solves f(x) + λ g(x)
    # ∇F(y) = ∇f(y)
    # ProxG(y) = \arg \minₓ 0.5 * || x - y ||_2^2 + λ g(x)
    # Supports in place ProxG

    λ *= η;

    for ii ∈ 1:numIterations
        # FISTA (Nesterov) Accelerated
    
        vG = ∇F(vZ);
    
        vW .= vX; #<! Previous iteration
        vX .= vZ .- (η .* vG);
        vX .= ProxG(vX, λ);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW));
    end

    return vX;

end

function ProximalGradientDescentAcc( vX :: AbstractVector{T}, ∇F :: Function, ProxFun :: Function, η :: T, numIterations :: S; λ :: T = one(T) ) where {T <: AbstractFloat, S <: Integer}

    vG = similar(vX);
    vZ = copy(vX);
    vW = similar(vX);
    vX = ProximalGradientDescentAcc!(vX, vG, vZ, vW, ∇F, ProxFun, η, numIterations; λ = λ);

    return vX;

end

function OrthogonalProjectionOntoConvexSets( vY :: AbstractVecOrMat{T}, vProjFun :: AbstractVector{<: Function}; numIter :: S = 1_000, ε :: T = T(1e-6) ) where {T <: AbstractFloat, S <: Integer}

    numSets = length(vProjFun);
    
    aZ = [zeros(T, size(vY)) for _ ∈ 1:numSets];
    aU = [zeros(T, size(vY)) for _ ∈ 1:numSets]; #<! TODO: Optimize vU (Probably single instance)
    vT = copy(vY);
    vX = copy(vY);

    for ii ∈ 1:numIter
        maxVal = zero(T);
        for jj ∈ 1:numSets
            # aU[jj]  = vProjFun[jj](vT + aZ[jj]);
            aU[jj] .= vT .+ aZ[jj];
            aU[jj]  = vProjFun[jj](aU[jj]);
            aZ[jj] .= vT .+ aZ[jj] .- aU[jj];

            vT .= aU[jj]

            maxI = @fastmath maximum(abs(x - y) for (x, y) ∈ zip(vX, aU[jj])); #<! maximum(abs.(vX .- aU[jj]));
            if (maxVal < maxI)
                maxVal = maxI;
            end
        end
        stopCond = maxVal < ε;

        vX .= vT;
        
        if stopCond
            break;
        end
    end

    return vX;

end

function DouglasRachford( mD :: Matrix{T}, vY :: VecOrMat{T}, hProxF :: Function, hProxG :: Function; numItr :: N = 500, γ :: T = T(1.0), λ :: T = T(1.0) ) where {T <: AbstractFloat, N <: Integer}
    # Solves with Douglas Rachford Splitting:
    # \arg \minₚ f(x) + g(x)
    # TODO: Find optimization for γ and λ

    for ii ∈ 1:numItr
        vX   = hProxG(vY);
        vY .+= λ * (hProxF(T(2) * vX - vY) - vP);
    end

    return vP;

end

function IRLS!( vX :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vW :: Vector{T}, mWA :: Matrix{T}, mC :: Matrix{T}, vT :: Vector{T}, sBKWorkSpace :: BunchKaufmanWs{T}; normP :: T = one(T), numItr :: N = UInt32(100), ϵ :: T = T(1e-6) ) where {T <: AbstractFloat, N <: Unsigned}
    # Solves ||A * x - b||ₚ
    # TODO: Optimize for the case m >> n.

    errThr = T(1e-6); #<! Should be adaptive per iteration
    effNorm = ((normP - T(2)) / T(2)); #<! Solving the Normal Equations (Doubling vW -> Power divided by 2)
    
    for _ in 1:numItr
        mul!(vW, mA, vX);
        vW .-= vB; #<! Error
        # Basically solving (vW .* A) \ (vW .* vB) <-> (mA' * Diag(vW) * mA) \ (mA' * Diag(vW) * vB).
        # Assuming m << n (size(mA, 1) << size(mA, 2)) it is faster to solve the normal equations.
        # The cost is doubling the condition number.
        vW .= max.(abs.(vW), errThr) .^ effNorm;
        vW .= vW ./ sum(vW);
        mWA .= vW .* mA;
        mul!(mC, mWA', mWA); #<! (mWA' * mWA) 
        # mC .= 0.5 .* (mC .+ mC'); #<! Guarantees symmetry (Allocates, seems to protect from aliasing)
        # for jj in 2:size(mC, 2)
        #     for ii in 1:(jj - 1)
        #         mC[jj, ii] = mC[ii, jj];
        #     end
        # end
        # No need to symmetrize `mC` as the decomposition looks only on a single triangle
        vW .= vW .* vB; #<! (mW * vB);
        copy!(vT, vX); #<! Previous iteration
        mul!(vX, mWA', vW); #<! (mWA' * mW * vB);
        # ldiv!(cholesky!(mC), vX); #<! vX = (mWA' * mWA) \ (mWA' * mW * vB);
        # Using Bunch-Kaufman as it works for SPSD (Cholesky requires SPD).
        _, ipiv, _ = LAPACK.sytrf!(sBkWorkSpace, 'U', mC); #<! Applies the decomposition
        sBkFac = BunchKaufman(mC, ipiv, 'U', true, false, BLAS.BlasInt(0));
        ldiv!(sBkFac, vX); #<! vX = (mWA' * mWA) \ (mWA' * mW * vB);
        vT .= abs.(vX .- vT);
        if maximum(vT) <= ϵ
            break;
        end
    end

    return vX;
    
end

function IRLS( mA :: Matrix{T}, vB :: Vector{T}; normP :: T = one(T), numItr :: N = UInt32(100) ) where {T <: AbstractFloat, N <: Unsigned}

    vX  = Vector{T}(undef, size(mA, 2));
    vT  = Vector{T}(undef, size(mA, 2));
    vW  = Vector{T}(undef, size(mA, 1));
    mWA = Matrix{T}(undef, size(mA));
    mC  = Matrix{T}(undef, size(mA, 2), size(mA, 2));
    sBkWorkSpace = BunchKaufmanWs(mC);

    vX = IRLS!(vX, mA, vB, vW, mWA, mC, vT, sBkWorkSpace; normP = normP, numItr = numItr);

    return vX;
    
end

function FindZeroBinarySearch( hF :: Function, valA :: T, valB :: T; tolValue :: T = T(1e-8), maxIter :: N = 100 ) where {T <: AbstractFloat, N <: Integer}
    
    valFA = hF(valA);
    valFB = hF(valB);

    if valFA * valFB > zero(T)
        error("Function must have opposite signs at valA and valB: f(valA) = $valFA, f(valB) = $valFB");
    end

    for ii in 1:maxIter
        midPoint = (valA + valB) / T(2);
        valFMid  = hF(midPoint);

        if (abs(valFMid) < tolValue) || (((valB - valA) / T(2)) < tolValue)
            return midPoint;
        end

        if valFA * valFMid < zero(T)
            valB = midPoint;
            valFB = valFMid;
        else
            valA = midPoint;
            valFA = valFMid;
        end
    end

    return (valA + valB) /  T(2);

end

function LsqBox( mA :: AbstractMatrix{T}, vB :: AbstractVector{T}, vL :: AbstractVector{T}, vU :: AbstractVector{T}; numIter :: N = 100, relTol :: T = T(-1), absTol :: T = T(0)) where {T <: AbstractFloat, N <: Integer}
    # Based on Steven G. Johnson code (See https://discourse.julialang.org/t/35611/13)
    # Similar to Dimitri P. Bertsekas - Projected Newton Methods for Optimization Problems with Simple Constraints (https://ieeexplore.ieee.org/document/4047042).
    
    Base.require_one_based_indexing(mA, vB, vL, vU);
    numRows = size(mA, 1);
    numCols = size(mA, 2);
    
    if relTol < zero(T)
        # Default
        relTol = eps(T) * sqrt(T(numCols));
    end
    
    vX = mA \ vB;
    @. vX = clamp(vX, vL, vU); #<! Make a feasible candidate
    # Pre calculation
    AᵀA = mA' * mA;
    Aᵀb = mA' * vB;
    # Gradient ∇ₓ of ½‖Ax - b‖²
    vG = AᵀA*vX;
    vG .-= Aᵀb;
    # Inactive constraints mask
    vInactive = Bool[vL[i] < vU[i] && (vX[i] != vL[i] || vG[i] ≤ T(0)) && (vX[i] != vU[i] || vG[i] ≥ T(0)) for i in eachindex(vX)];
    all(vInactive) && return vX, true; #<! All constraints are inactive, the unconstrained solution is valid
    vActive = map(!, vInactive);
    vXPrev  = copy(vX);
    for ii in 1:numIter
        vXa = mA[:, vInactive] \ (vB - mA[:, vActive] * vX[vActive]); #<! Working on the residual
        vX[vInactive] = vXa;
        @. vX = clamp(vX, vL, vU);
        # Update the gradient
        mul!(vG, AᵀA, vX);
        vG .-=  Aᵀb;
        # Update the inactive mask
        for jj in eachindex(vX)
            vInactive[jj] = vL[jj] < vU[jj] && (vX[jj] != vL[jj] || vG[jj] ≤ T(0)) && (vX[jj] != vU[jj] || vG[jj] ≥ T(0));
        end
        # Check for convergence
        all(i -> vInactive[i] == !vActive[i], eachindex(vActive)) && return vX, true; #<! Convergence: active set unchanged 
        norm(vX - vXPrev) ≤ max(relTol * norm(vX), absTol) && return vX, true; #<! Convergence: `vX` not changing much
        vXPrev .= vX; #<! Keep current candidate as a previous for next step
        @. vActive = !vInactive; #<! Update active mask
    end
    
    # No convergence
    return vX, false;

end