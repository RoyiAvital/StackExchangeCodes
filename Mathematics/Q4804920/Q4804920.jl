# StackExchange Mathematics Q4804920
# https://math.stackexchange.com/questions/4804920
# Optimize Summation of L2 Norm and Infinity Norm.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     16/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using LinearAlgebra;
using MAT;
using PlotlyJS;


## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = false;

dUtfSymPx   = Dict(UInt8(0) => '🟩', UInt8(128) => '🟦', UInt8(255) => '🟥');
dUtfSymBool = Dict(false => '🟥', true => '🟩');
dUtfSymDir  = Dict(Int8(-1) => '↖', Int8(0) => '↑', Int8(1) => '↗');

## Functions

function MatInv!( mA :: Matrix{<: LinearAlgebra.BlasReal} )
    # Using `L` means the output is on the lower triangle.
    # The upper triangle is a buffer for calculations.
    _, info = LAPACK.potrf!('L', mA);
    (info == 0) || throw(PosDefException(info));
    LAPACK.potri!('L', mA); #<! Returns only a Triangle of the inverse

    for ii in 1:size(mA, 1)
        for jj in (ii + 1):size(mA, 2)
            mA[ii, jj] = mA[jj, ii];
        end
    end

    return mA
end

function SolveL2LInfAdmm!( vX :: Vector{T}, mX :: Matrix{T}, mA :: Matrix{T}, vY :: Vector{T}, λ :: T, ρ :: T, numIterations :: N, mT :: Matrix{T}, vU :: Vector{T}, vZ :: Vector{T}, vT1 :: Vector{T}, vT2 :: Vector{T} ) where {T <: AbstractFloat, N <: Unsigned}

    numElements = length(vX);
    fill!(mT, zero(T));
    for ii in 1:size(mT, 1)
        mT[ii, ii] = T(1.0);
    end

    mul!(mT, mA', mA, ρ, T(1.0));
    MatInv!(mT);

    mX[:, 1] .= vX;

    for ii in 2:numIterations
        # Calculate vX
        @. vT1 = vZ + vY - vU;
        mul!(vT2, mA', vT1);
        mul!(vX, mT, vT2, ρ, T(0.0));

        # Calculate vZ
        mul!(vT1, mA, vX);
        @. vT1 -= (vY - vU);
        ProxLInf!(vZ, vT1, λ / ρ);

        # Calculate vU
        mul!(vT2, mA, vX);
        @. vU += vT2 - vZ - vY;

        mX[:, ii] .= vX;
    end

end

function ProxLInf!(vO :: Vector{T}, vX :: Vector{T}, λ :: T) where {T <: AbstractFloat}

    vX ./= λ;
    ProjectL1Ball!(vO, vX, T(1.0));
    # vO .= vX .- λ .* vO;
    @. vO = λ * (vX - vO);
    
end

function ProjectL1Ball!(vX :: Vector{T}, vY :: Vector{T}, ballRadius :: T) where {T <: AbstractFloat}

    numElements = length(vY);
    
    if(sum(abs, vY) <= ballRadius)
        # The input is already within the L1 Ball.
        vX .= vY;
        return
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

        if (objVal < 0)
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
    
    @. vX = sign(vY) * max(abs(vY) - λ, 0);

end

function ProjectL1BallObj( vZ :: Vector{T}, λ :: T, ballRadius :: T ) where {T <: AbstractFloat}

    objVal = zero(T);
    for ii in 1:length(vY)
        objVal += max(vZ[ii] - λ, 0);
    end

    objVal -= ballRadius;

    return objVal;
    
end

## Test

# vY = [1.2, 0.9, 1.4, -1.9];
# vX = zeros(length(vY));
# ProjectL1Ball!(vX, vY, 1.0);

## Parameters

# Problem parameters
matFileName = "Data.mat";
λ           = 0.7;

# Solver Parameters
ρ               = 3.0;
numIterations   = Unsigned(100);

#%% Load / Generate Data
dData = matread(matFileName);
vA = dData["vA"][:];
vY = dData["vY"][:];

optVal = dData["optVal"][1]; #<! Optimal value by CVX

numElements = length(vA);

vX = zeros(numElements);
mX = zeros(numElements, numIterations);
mA = diagm(vA);
vU = zeros(size(mA, 1));
vZ = zeros(size(mA, 1));

# Buffers
mT  = zeros(numElements, numElements);
vT1 = zeros(size(mA, 1));
vT2 = zeros(numElements);

# Should be paramLambda * max(mA * vX - vY) where mA is diagonal
hObjFun( vX :: Vector{<: AbstractFloat} ) = 0.5 * sum(abs2, vX) + λ * maximum(abs, vA .* vX .- vY);

## Analysis
SolveL2LInfAdmm!(vX, mX, mA, vY, λ, ρ, numIterations, mT, vU, vZ, vT1, vT2);

vObjFun = [hObjFun(mX[:, ii]) for ii in 1:numIterations];

## Display Results

figureIdx += 1;

oTrace1 = scatter(x = 1:numIterations, y = vObjFun, mode = "lines", text = "ADMM", name = "ADMM",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = 1:numIterations, y = optVal * ones(numIterations), 
                  mode = "lines", text = "Optimal Value", name = "Optimum",
                  line = attr(width = 1.5, dash = "dot"));
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Value");
hP = plot([oTrace1, oTrace2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

