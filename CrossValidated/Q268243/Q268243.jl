# StackExchange Cross Validated Q268243
# https://stats.stackexchange.com/questions/268243
# Solve a Convex Objective with L2 Term and L1 Term.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     18/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( mX :: Matrix{T}, mA :: Matrix{T}, mB :: Matrix{T}, mC :: Matrix{T}, λ :: T ) where {T <: AbstractFloat}

    valObj = T(0.5) * sum(abs2, mX + mB) + λ * sum(abs, mA * mX - mC);

    return valObj;
    
end

function CVXSolver( mA :: Matrix{T}, mB :: Matrix{T}, mC :: Matrix{T}, λ :: T ) where {T <: AbstractFloat}

    numRows = size(mB, 1);
    numCols = size(mB, 2);
    mX      = Convex.Variable(numRows, numCols);
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = minimize( T(0.5) * Convex.sumsquares(mX + mB) + λ * Convex.norm_1(mA * mX - mC) );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return mX.value;

end

function ADMM!(tX :: Array{T, 3}, mZ :: Matrix{T}, mU :: Matrix{T}, mA :: Matrix{T}, mC :: Matrix{T}, hProxF :: Function, hProxG :: Function; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat}
    # Solves f(X) + λ g(A * X - C)
    # Where Z = A X - C, and g(Z) has a well defined Prox.
    # ADMM for the case A X + Z = C
    # ProxF(Y) = \arg \minₓ 0.5ρ * || A X - Y - C + U ||_2^2 + f(x)
    # ProxG(Y) = \arg \minₓ 0.5ρ * || A X - Y - C + U ||_2^2 + λ g(x)
    # Initialization by mX[:, 1]
    # Supports in place ProxG

    numIterations = size(tX, 3);
    
    for ii ∈ 2:numIterations
        mX = @view tX[:, :, ii];

        mX .= hProxF(mZ + mC - mU, ρ);
        mul!(mZ, mA, mX);
        mZ .+= (mU .- mC);
        mZ .= hProxG(mZ, λ / ρ);
        # mZ .= hProxG(mA * mX + mU - mC, λ / ρ);
        mU += mA * mX - mZ - mC;
    end

end


## Parameters

# Data
numRowsA = 30;
numColsA = 27;

numRowsX = numColsA;
numColsX = 18;

# Model
λ = 0.25;

# Solver
numIterations = 100;
ρ = 1.0;

## Load / Generate Data

mA = randn(oRng, numRowsA, numColsA);
mB = randn(oRng, numRowsX, numColsX); #<! Same as `mX`
mC = randn(oRng, numRowsA, numColsX);

hObjFun(mX :: Matrix{T}) where {T <: AbstractFloat} = ObjFun(mX, mA, mB, mC, λ);

dSolvers = Dict();

## Analysis
# Model: 0.5 * || X + B ||_F^2 + λ * || A * X - C ||_1

# DCP Solver
methodName = "Convex.jl"

mXRef = CVXSolver(mA, mB, mC, λ);
optVal = hObjFun(mXRef);

dSolvers[methodName] = optVal * ones(numIterations);

# Projected Gradient Descent
methodName = "ADMM";

mAA = mA' * mA;

hProxF(mY :: Matrix{T}, ρ :: T) where {T <: AbstractFloat} = (ρ * mAA + I) \ (ρ * mA' * mY - mB);
hProxG(mY :: Matrix{T}, γ :: T) where {T <: AbstractFloat} = sign.(mY) .* max.(abs.(mY) .- γ, zero(T));

tX = zeros(numRowsX, numColsX, numIterations);
mZ = zero(mC);
mU = zero(mC);

# Specific variant for Matrices
ADMM!(tX, mZ, mU, mA, mC, hProxF, hProxG; ρ = ρ, λ = λ);

dSolvers[methodName] = [hObjFun(tX[:, :, ii]) for ii ∈ 1:size(tX, 3)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end