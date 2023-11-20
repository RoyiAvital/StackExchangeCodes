# StackExchange Mathematics Q3042354
# https://math.stackexchange.com/questions/3042354
# Large Scale Quadratic Form with Linear Box Constraints.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  The condition number impacts the performance of PD3O and Chambolle significantly.
#       Hence some conditioning is added to the generation of `mA`.
#   3. 
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     18/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
using Random;
# External
using Convex;
using MAT;
using LinearAlgebra;
using Optim;
using PlotlyJS;
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)
# Random.default_rng() = StableRNG(RNG_SEED); #<! Danger! This is a hack.

## General Parameters

figureIdx = 0;

exportFigures = false;

dUtfSymPx   = Dict(UInt8(0) => '🟩', UInt8(128) => '🟦', UInt8(255) => '🟥');
dUtfSymBool = Dict(false => '🟥', true => '🟩');
dUtfSymDir  = Dict(Int8(-1) => '↖', Int8(0) => '↑', Int8(1) => '↗');

## Functions

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
        # vS .= vS + δ * mA * vX̄;
        vS .= vS + mA * vX̄;
        
        # vS .= vS - δ * hProxH(μ * vS, μ);
        vS .= vS - hProxH(vS, μ); #<! Prox of Conjugate (https://github.com/mingyan08/PD3O/issues/2)
        vX = @view mX[:, ii];
        vX .= hProxG(vT - λ * mA' * vS, γ);
        vX̄ .= 2vX - vT - γ * h∇f(vX);
    end

end

function ChamPock!(mX :: Matrix{T}, vP :: Vector{T}, vX̄ :: Vector{T}, hProxF⁺ :: Function, hProxG :: Function, σ :: T, τ :: T; θ :: T = 1.0) where {T <: AbstractFloat, F <: Function}

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        vT = @view mX[:, ii - 1];; #<! Previous iteration
        vX = @view mX[:, ii];
        
        vP .= hProxF⁺(vP + (σ * mA * vX̄), σ);
        vX .= hProxG(vT - (τ * mA' * vP), τ);
        
        vX̄ .= vX + (θ * (vX - vT));
    end

end


## Parameters

# Data
numRows = 5;
numCols = numRows; #<! PSD Matrix
valA    = 0.23;
valB    = 1.05;
δ       = 1e6;
valTol  = 1e-3;

loadData = false;

# Solvers
numIterations = 2500;

# ADMM Solver
ρ = 2.5;

# PD3O

# Chamoblle Pock

## Generate / Load Data
oRng = StableRNG(1234);
mA = randn(oRng, numRows, numCols);
mA = mA' * mA;
mA = mA + 0.1I; #<! High condition numbers makes convergence slower
mA = mA + mA';

if (loadData)
    dVars = matread("Data.mat");
    subStreamNumber = dVars["subStreamNumber"];
    mA = dVars["mA"];
    numRows, numCols = size(mA);
end

mX = zeros(numCols, numIterations);
vX = vX = mA \ (((valA + valB) / 2) * ones(numCols));
mX[:, 1] = vX;

hδFun( vX :: Vector{<: AbstractFloat} ) = δ * (any((mA * vX) .> (valB + valTol) .|| (mA * vX) .< (valA - valTol)));
hObjFun( vX :: Vector{<: AbstractFloat} ) = 0.5 * dot(vX, mA, vX) + hδFun(vX);

dSolvers = Dict();

## Analysis

# DCP Solver
vX0 = Variable(numCols);
sConvProb = minimize(0.5 * quadform(vX0, mA; assume_psd = true), (mA * vX0) <= valB, (mA * vX0) >= valA);
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
vXRef = vX0.value
optVal = sConvProb.optval;

if (loadData)
    optVal = dVars["optVal"];
    vXRef = dVars["vXRef"];
end


# ADMM
# Solves: f(x) + g(z) subject to Px + Qz + r = 0
# f(x) = 0.5 * x' * A * x -> Prox_f(y) = (λ * A' * A + A) \ (λ * A' * y)
# g(x) = δ(A x) ∈ [a, b] -> Prox_g(y) = clamp(y, a, b)
# P = A, Q = -I, r = 0
methodName = "ADMM";

sEigFac = eigen(mA);

hD( λ :: T ) where {T <: AbstractFloat} = (λ .* sEigFac.values) ./ ((λ .* (sEigFac.values .^ 2)) .+ sEigFac.values);
hProxF( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = sEigFac.vectors * (hD(λ) .* (sEigFac.vectors' * vY));
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);

vZ = mA * mX[:, 1];
vU = zeros(numCols);

ADMM!(mX, vZ, vU, hProxF, hProxG; ρ = ρ);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


# PD3O
# Solves: \arg \min_x f(x) + g(x) + h(A * x)
# f(x) = 0.5 * x' * A * x
# g(x) = 0 -> Prox_g(y) = y
# h(x) = δ(A x) ∈ [a, b] -> Prox_h(y) = clamp(y, a, b)
# Useful as it doesn't require solving big linear equation.

methodName = "PD3O";

valL = opnorm(mA)
β = 1 / valL;
γ = 1.8β;
λ = 0.9β * β; #<! γ * δ
μ = γ / λ; #<! In the paper 1/δ
δ = λ / γ;

h∇f( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = mA * vX;
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vY;
hProxH( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);

vX̄ = copy(mX[:, 1]);
vS = mA * mX[:, 1];
vT = zeros(numCols);

PD3O!(mX, vS, vX̄, vT, h∇f, hProxG, hProxH, γ, λ);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


# Dual Prox
# Solves: arg min_x f(A * x) + g(x)
# f(A * x) = δ(A x) ∈ [a, b] -> Prox_h(y) = clamp(y, a, b)
# g(x) = x' * A * x
# The Prox of g(x) will require solving a Linear System equation.

methodName = "ChambollePock";

valL = opnorm(mA' * mA);

τ = sqrt(1 / (1.05 * valL));
σ = sqrt(1 / (1.05 * valL));
θ = 1.0;

hProxF( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = clamp.(vY, valA, valB);
hProxF⁺( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vY - λ * hProxF(vY ./ λ, 1 / λ); #<! Prox of conjugate
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = (λ * mA + I) \ vY;

vP = mX[:, 1];
vX̄ = mX[:, 1];

ChamPock!(mX, vP, vX̄, hProxF⁺, hProxG, σ, τ; θ = θ)

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end
