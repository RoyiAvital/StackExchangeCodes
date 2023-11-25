# StackOverflow Q35813091
# https://stackoverflow.com/questions/35813091
# Optimize the Trace with PSD and Trace Constraints.
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
# - 1.0.000     25/11/2023  Royi Avital
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

function ProjSymMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    return (mX' .+ mX) ./ T{2.0};
end

function ProjPsdMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    sEigFac = eigen(mX);
    return sEigFac.vectors * diagm(max.(sEigFac.values, 0)) * sEigFac.vectors';
end

function ProjUniTrMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    return mX .- (((tr(mX) - one(T)) / size(mX, 1)) * I);
end

function ProjSetConDykstra( mU :: Matrix{T}, mZ :: Matrix{T}, vProjFun :: Vector{<: Function}, numIterations :: N ) where {T <: AbstractFloat, N <: Integer}

    numSets = size(mU, 2);

    for kk ∈ 1:numIterations
        mU[:, 1] .= vProjFun[ii](mU[:, end] .+ mZ[:, 1]);
        mZ[:, 1] .= mU[:, end] .+ mZ[:, 1] .- mU[:, 1];
        for ii ∈ 2:numSets
            mU[:, ii] .= vProjFun[ii](mZ[:, ii - 1]);
            mZ[:, ii] .= mU[:, ii - 1] .+ mZ[:, ii] .- mU[:, ii];
        end 
    end

end

function ADMM!(mX :: Matrix{T}, vZ :: Vector{T}, vU :: Vector{T}, hProxF :: Function, hProxG :: Function; ρ :: T = 2.5) where {T <: AbstractFloat, F <: Function}

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        vX = @view mX[:, ii];

        vX .= hProxF(vZ - vU, ρ);
        vZ .= hProxG(mA * vX + vU, 1 / ρ);
        vU .= vU + mA * vX - vZ;
    end

end

function PD3O!(mX :: Matrix{T}, vS :: Vector{T}, vX̄ :: Vector{T}, vT :: Vector{T}, h∇f :: Function, hProxG :: Function, hProxH :: Function, γ :: T, λ :: T) where {T <: AbstractFloat, F <: Function}

    numIterations = size(mX, 2);
    δ = λ / γ;
    μ = γ / λ; #<! (1 / δ)
    
    vX = @view mX[:, 1];
    
    for ii ∈ 2:numIterations
        vT .= vX - γ * h∇f(vX); #<! Buffer (vXH)
        
        vS .= vS + δ * mA * vX̄;
        vS .= vS - δ * hProxH(μ * vS, μ);
        vX = @view mX[:, ii];
        vX .= hProxG(vT - γ * mA' * vS, γ);
        vX̄ .= 2vX - vT - γ * h∇f(vX);
    end

end


## Parameters

# Data
numRows = 5;
numCols = numRows; #<! Symmetric Matrix
valA    = 0.23;
valB    = 1.05;
δ       = 1e6;
δTol    = 1e-5;

# Solvers
numIterations = 25000;

# Projected Gradient Descent

# ADMM Solver
ρ = 2.5;

# PD3O

# Chamoblle Pock

## Generate / Load Data
oRng = StableRNG(1234);
mB = randn(oRng, numRows, numCols);
# Will make `mB` SPSD to abs() in the DCP will make sense (https://math.stackexchange.com/questions/888677).
mB = mB' * mB + 0.1I;
mB = mB + mB';

# sEigFac = eigen(mA); #<! Needed for efficient solution of the linear system

# mX = zeros(numCols, numIterations);
# vX = vX = mA \ (((valA + valB) / 2) * ones(numCols));
# mX[:, 1] = vX;

# See https://discourse.julialang.org/t/73206
hδFun( mX :: Matrix{<: AbstractFloat} ) = δ * !(isapprox(tr(mX), 1; atol = δTol) && isapprox(mX, mX'; atol = δTol) && (eigmin(mX) > -δTol));
hObjFun( mX :: Matrix{<: AbstractFloat} ) = tr(mX * mB) + hδFun(mX);

dSolvers = Dict();

## Analysis

# DCP Solver
mX0 = Variable(numRows, numCols);
# Since mX0 and mB are SPSD the `tr()` in non negative.
# Hence one could use `abs()` to avoid complex numbers.
sConvProb = minimize(abs(tr(mX0 * mB)), mX0 == mX0', tr(mX0) == 1, mX0 in :SDP);
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
mXRef = mX0.value
optVal = sConvProb.optval;

# Projected Gradient Descent
vProjFun = [ProjSymMatSet, ProjPsdMatSet, ProjUniTrMatSet];
mU = zeros(numRows * numCols, length(vProjFun));
mZ = zeros(numRows * numCols, length(vProjFun));
hUpMat(mU :: Matrix{T}, vY :: Vector{T}) {T <: AbstractFloat} = mU;
∇F( vY :: Vector{T} ) where {T <: AbstractFloat} = vec(mB);
hProjFun( vY :: Vector{T} ) where {T <: AbstractFloat} = ProjSetConDykstra(hUpMat!(mU, vY), mZ, 2000);

mX = zeros(numRows * numCols, numIterations);

for ii ∈ 2:numIterations
    mX[:, ii] = mX[:, ii - 1] .- η * ∇F(mX[:, ii - 1]);
    mX[:, ii] = hProjFun(mX[:, ii]);
end


# ADMM
# Solves: f(x) + g(z) subject to Px + Qz + r = 0
# f(x) = 0.5 * x' * A * x -> Prox_f(y) = (λ * A' * A + A) \ (λ * A' * y)
# g(x) = δ(A x) ∈ [a, b] -> Prox_g(y) = clamp(y, a, b)
# P = A, Q = -I, r = 0
# methodName = "ADMM";

# hD( λ :: T ) where {T <: AbstractFloat} = (λ .* sEigFac.values) ./ ((λ .* (sEigFac.values .^ 2)) .+ sEigFac.values);
# hProxF( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = sEigFac.vectors * (hD(λ) .* (sEigFac.vectors' * vY));
# hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);

# vZ = mA * mX[:, 1];
# vU = zeros(numCols);

# ADMM!(mX, vZ, vU, hProxF, hProxG; ρ = ρ);

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


# PD3O
# Solves: \arg \min_x f(x) + g(x) + h(A * x)
# f(x) = 0.5 * x' * A * x
# g(x) = 0 -> Prox_g(y) = y
# h(x) = δ(A x) ∈ [a, b] -> Prox_h(y) = clamp(y, a, b)
# Useful as it doesn't require solving big linear equation.
# γ - Primal step size, δ - Dual step size.

# methodName = "PD3O";

# valL = opnorm(mA)
# β = 1 / valL;
# γ = 1.8β; #<! γ < 2β (Like a primal step size for Gradient Descent)
# λ = 0.9β * β; #<! γ * δ < β²
# μ = γ / λ; #<! In the paper 1/δ
# δ = λ / γ;

# h∇f( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = mA * vX;
# hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vY;
# hProxH( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);

# vX̄ = copy(mX[:, 1]);
# vS = mA * mX[:, 1];
# vT = zeros(numCols);

# PD3O!(mX, vS, vX̄, vT, h∇f, hProxG, hProxH, γ, λ);

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


# Dual Prox
# Solves: arg min_x f(A * x) + g(x)
# f(A * x) = δ(A x) ∈ [a, b] -> Prox_h(y) = clamp(y, a, b)
# g(x) = x' * A * x
# The Prox of g(x) will require solving a Linear System equation.

# methodName = "ChambollePock";

# valL = opnorm(mA' * mA);

# τ = sqrt(1 / (1.05 * valL));
# σ = sqrt(1 / (1.05 * valL));
# θ = 1.0;

# hProxF( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);
# hProxF⁺( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vY - λ * hProxF(vY ./ λ, 1 / λ); #<! Prox of conjugate
# hD( λ :: T ) where {T <: AbstractFloat} = 1 ./ ((λ .* sEigFac.values) .+ 1);
# hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = sEigFac.vectors * (hD(λ) .* (sEigFac.vectors' * vY));
# # hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = (λ * mA + I) \ vY;

# vP = mX[:, 1];
# vX̄ = mX[:, 1];

# ChamPock!(mX, vP, vX̄, hProxF⁺, hProxG, σ, τ; θ = θ)

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

# figureIdx += 1;

# vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# # shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
# for (ii, methodName) in enumerate(keys(dSolvers))
#     vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
#                mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
# end
# oLayout = Layout(title = "Objective Function, Condition Number = $(cond(mA))", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

# hP = plot(vTr, oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end
