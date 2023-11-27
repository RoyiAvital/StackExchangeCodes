# StackOverflow Q28768668
# https://stackoverflow.com/questions/28768668
# Minimize Total Variation Like L1 Norm Regularized Least Squares.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   3. 
# TODO:
# 	1.  C
# Release Notes
# - 1.0.000     26/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
using MAT;
using PlotlyJS;
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions

function SoftThreshold!( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # Soft Threshold - L1 Prox
    # See https://math.stackexchange.com/questions/1961888

    vY .= sign.(vY) .* (max.(abs.(vY) .- λ, zero(T)));
    return vY;


end

function ADMM!(mX :: Matrix{T}, vZ :: Vector{T}, vU :: Vector{T}, mA :: Matrix{T}, hProxF :: Function, hProxG :: Function; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat}
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

        vX .= hProxF(vZ - vU, ρ);
        mul!(vZ, mA, vX);
        vZ .+= vU;
        vZ .= hProxG(vZ, λ / ρ);
        # vZ .= hProxG(mA * vX + vU, λ / ρ);
        vU .= vU + mA * vX - vZ;
    end

end


## Parameters

# Data
numRows = 40;
numCols = 50;
λ       = 0.15;

# Solvers
numIterations = 1_000;

# ADMM
ρ = 0.5;

## Generate / Load Data
oRng = StableRNG(1234);
mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);
mD = randn(oRng, numRows, numCols);

hObjFun( vX :: Vector{<: AbstractFloat} ) = 0.5 * sum(abs2, mA * vX - vB) + λ * sum(abs, mD * vX);

dSolvers = Dict();


## Analysis

# DCP Solver
vX0 = Variable(numCols);
sConvProb = minimize(0.5 * sumsquares(mA * vX0 - vB) + λ * norm(mD * vX0, 1));
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
vXRef  = vec(vX0.value);
optVal = sConvProb.optval;

# Projected Gradient Descent (Proximal Gradient Descent)
methodName = "ADMM";

sCholFac = cholesky(mA' * mA + ρ * mD' * mD);
vAb      = mA' * vB;

hProxF( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = sCholFac \ (vAb + (λ * mD' * vY));
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = SoftThreshold!(vY, λ);

mX = zeros(numCols, numIterations);
mX[:, 1] .= zeros(numCols)

vZ = zeros(numRows);
vU = zeros(numRows);

ADMM!(mX, vZ, vU, mD, hProxF, hProxG; ρ = ρ, λ = λ);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function, Condition Number = $(@sprintf("%0.3f", cond(mA)))", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end