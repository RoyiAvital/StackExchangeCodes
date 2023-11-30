# StackExchange Mathematics Q3800282
# https://math.stackexchange.com/questions/3800282
# Solve Composition of Linear Least Squares with L2 and L2 Squared Regularization.
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
# - 1.0.000     29/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
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

function PD3O!(mX :: Matrix{T}, vS :: Vector{T}, vX̄ :: Vector{T}, vT :: Vector{T}, hA :: Function, hAᵀ :: Function, h∇f :: Function, hProxG :: Function, hProxH :: Function, γ :: T, λ :: T; α₁ :: T = one(T), α₂ :: T = one(T)) where {T <: AbstractFloat}
    # Solves: \arg \min_x f(x) + g(x) + h(A * x)
    # α₁, α₂: Prox constants of g() and h().
    # γ - Primal step size, δ - Dual step size.

    numIterations = size(mX, 2);
    δ = λ / γ;
    μ = γ / λ; #<! (1 / δ)
    
    vX = @view mX[:, 1];
    
    for ii ∈ 2:numIterations
        vT .= vX - γ * h∇f(vX); #<! Buffer (vXH)
        
        vS .= vS + δ * hA(vX̄);
        vS .= vS - δ * hProxH(μ * vS, α₂ * μ);
        vX = @view mX[:, ii];
        vX .= hProxG(vT - γ * hAᵀ(vS), α₁ * γ);
        vX̄ .= 2vX - vT - γ * h∇f(vX);
    end

end


## Parameters

# Data
numRows = 500;
numCols = 400; #<! PSD Matrix

λ₁ = 0.2;
λ₂ = 1.5;

# Solvers
numIterations = 2500;

# PD3O


## Generate / Load Data
oRng = StableRNG(1234);
mA = randn(oRng, numRows, numCols);
vB = randn(oRng, numRows);

mX = zeros(numCols, numIterations);

hObjFun( vX :: Vector{<: AbstractFloat} ) = 0.5 * sum(abs2, mA * vX - vB) + λ₁ * norm(vX, 2) + 0.5 * λ₂ * sum(abs2, vX);

dSolvers = Dict();

## Analysis

# DCP Solver
vX0 = Variable(numCols);
sConvProb = minimize(0.5 * sumsquares(mA * vX0 - vB) + λ₁ * norm(vX0, 2) + 0.5 * λ₂ * sumsquares(vX0));
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
vXRef = vX0.value
optVal = sConvProb.optval;

# PD3O
# Solves: \arg \min_x f(x) + g(x) + h(P * x)
# f(x) = 0.5 * || A * x - b ||_2^2
# g(x) = λ₁ || x ||_2 -> Prox_g(y) = 
# h(x) = 0.5 * λ₂ || x ||_2^2 -> Prox_h(y) = 
# Useful as it doesn't require solving big linear equation.
# γ - Primal step size, δ - Dual step size.

methodName = "PD3O";

valL = 1; #<! Since h(P * x) = h(I * x) = h(x)
β = 1 / opnorm(mA' * mA); #<! By f(x)
γ = 1.8β; #<! γ < 2β (Like a primal step size for Gradient Descent)
λ = 10β * β; #<! γ * δ < (1 / (L * L))
μ = γ / λ; #<! In the paper 1/δ
δ = λ / γ;

hQ( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = vX; #<! P = I
hQᵀ( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = vX; #<! P = I
h∇f( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = mA' * (mA * vX - vB); #<! Can be optimized
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = (1 .- (λ ./ max(norm(vY, 2), λ))) * vY; #<! Can be done in place
hProxH( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = (1 / (1 + λ)) .* vY; #!< Can be done inplace

vX̄ = copy(mX[:, 1]);
vS = mX[:, 1];
vT = zeros(numCols);

PD3O!(mX, vS, vX̄, vT, hQ, hQᵀ, h∇f, hProxG, hProxH, γ, λ; α₁ = λ₁, α₂ = λ₂);

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
