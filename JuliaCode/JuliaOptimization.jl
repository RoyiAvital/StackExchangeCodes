# StackExchange Code - Julia Optimization
# Set of functions for Optimization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
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

# External


## Constants & Configuration
include("./JuliaInit.jl");

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

    # Using actual function defintion (Equivaklent to the anonymous above).
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

function GradientDescentBackTracking!( vX :: AbstractVecOrMat{T}, vG :: AbstractVecOrMat{T}, vZ :: AbstractVecOrMat{T}, numIter :: S, η :: T, ObjFun :: Function, ∇ObjFun :: Function; ProjFun :: Function = identity, α :: T = T(0.5), β :: T = T(1e-10) ) where {T <: AbstractFloat, S <: Integer}
    # Non allocating assuming function are not allocating.

    for ii ∈ 1:numIter
        vG .= ∇ObjFun(vX);
        objFunVal = ObjFun(vX);
        vZ .= vX .- η .*  vG;
        while ((hObjFun(vZ) > objFunVal) && (α > β))
            η *= α;
            vZ .= vX .- η .*  vG;
        end
        vG .*= η;
        η   /= α;
        vX .-= vG;
    
        vX .= mX .- (η .* ∇vX);
        vX = ProjFun(vX);
    end

end

function GradientDescentBackTracking( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ObjFun :: Function, ∇ObjFun :: Function; ProjFun :: Function = identity, α :: T = T(0.5), β :: T = T(1e-10) ) where {T <: AbstractFloat, S <: Integer}
    # Mutates vX

    vG = copy(vX);
    vZ = copy(vX);

    GradientDescentBackTracking(vX, vG, vZ, numIter, η, ObjFun, ∇ObjFun; ProjFun = ProjFun, α = α, β = β);

end



function GradientDescentAccelerated( vX :: AbstractVecOrMat{T}, numIter :: S, η :: T, ∇ObjFun :: G; ProjFun :: F = identity ) where {T <: AbstractFloat, S <: Integer, F <: Function, G <: Function}
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

function ADMM!(mX :: Matrix{T}, vZ :: Vector{T}, vU :: Vector{T}, mA :: AbstractMatrix{T}, hProxF :: Function, hProxG :: Function; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat}
    # Solves f(x) + λ g(Ax)
    # Where z = Ax, and g(z) has a well defined Prox.
    # ADMM for the case Ax + z = 0
    # ProxF(y) = \arg \minₓ 0.5ρ * || A x - y ||_2^2 + f(x)
    # ProxG(y) = \arg \minₓ 0.5ρ * || x - y ||_2^2 + λ g(x)
    # Initialization by mX[:, 1]
    # Supports in place ProxG

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        vX = @view mX[:, ii];

        vZ .-= vU;
        vX .= hProxF(vZ, ρ);
        mul!(vZ, mA, vX);
        vZ .+= vU;
        vZ .= hProxG(vZ, λ / ρ);
        # vX .= hProxF(vZ - vU, ρ);
        # vZ .= hProxG(mA * vX + vU, λ / ρ);
        vU .= vU + mA * vX - vZ;
    end

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

function IRLS!( vX :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vW :: Vector{T}, mWA :: Matrix{T}, mC :: Matrix{T}, vT :: Vector{T}, sBKWorkSpace :: BunchKaufmanWs{T}; normP :: T = one(T), numItr :: N = 1000, ϵ :: T = T(1e-6) ) where {T <: AbstractFloat, N <: Unsigned}

    errThr = T(1e-6); #<! Should be adaptive per iteration
    effNorm = ((normP - T(2)) / T(2));
    
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

function IRLS(mA :: Matrix{T}, vB :: Vector{T}; normP :: T = one(T), numItr :: N = 1000 ) where {T <: AbstractFloat, N <: Unsigned}

    vX  = Vector{T}(undef, size(mA, 2));
    vT  = Vector{T}(undef, size(mA, 2));
    vW  = Vector{T}(undef, size(mA, 1));
    mWA = Matrix{T}(undef, size(mA));
    mC  = Matrix{T}(undef, size(mA, 2), size(mA, 2));
    sBkWorkSpace = BunchKaufmanWs(mC);

    vX = IRLS!(vX, mA, vB, vW, mWA, mC, vT, sBkWorkSpace; normP = normP, numItr = numItr);

    return vX;
    
end

