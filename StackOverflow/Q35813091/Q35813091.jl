# StackOverflow Q35813091
# https://stackoverflow.com/questions/35813091
# Optimization of a Trace Operator with Symmetric, PSD and Unit Trace Constraints.
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
using FastLapackInterface;
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

function hUpMat!( mU :: Matrix{T}, vY :: Vector{T}, arrIdx :: N ) where {T <: AbstractFloat, N <: Integer}
    mU[:, arrIdx] .= vY;
    return mU;
end

function ProjSymMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    # `mX` assumed to be square
    return (mX' .+ mX) ./ T(2.0);
end

function ProjPsdMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    # `mX` assumed to be symmetric
    sEigFac = eigen(mX);
    return sEigFac.vectors * diagm(max.(sEigFac.values, zero(T))) * sEigFac.vectors';
end

function ProjUniTrMatSet( mX :: Matrix{T} ) where {T <: AbstractFloat}
    # `mX` assumed to be square
    return mX .- (((tr(mX) - one(T)) / size(mX, 1)) * I(size(mX, 1)));
end

function ProjUniTrMatSet!( mX :: Matrix{T} ) where {T <: AbstractFloat}
    # `mX` assumed to be square
    mX .-= (((tr(mX) - one(T)) / size(mX, 1)) * I(size(mX, 1)));
end

function ProjSPSDMatSet!( mX :: Matrix{T}; numIter :: N = 1000, δ :: T = 1e-6 ) where {T <: AbstractFloat, N <: Integer}
    # `mX` assumed to be a square matrix
    sEigWs = EigenWs(mX, rvecs = true);

    mY = Matrix{T}(undef, size(mX)); #<! Previous iteration
    mT = Matrix{T}(undef, size(mX)); #<! Buffer

    for ii ∈ 1:numIter
        copy!(mY, mX);
        mX .= (mX' .+ mX) ./ T(2.0);
        sLapEig = LAPACK.geevx!(sEigWs, 'N', 'N', 'V', 'N', mX);
        sEigFac = LinearAlgebra.Eigen(sLapEig[2], sLapEig[5]);
        sEigFac.values .= max.(sEigFac.values, zero(T));
        mT .= sEigFac.values .* sEigFac.vectors';
        mul!(mX, sEigFac.vectors, mT);
        mY .= abs.(mY .- mX);
        if (maximum(mY) < δ)
            break;
        end
    end

    return mX;
    
end

function ProjSetConDykstra( mU :: Matrix{T}, mZ :: Matrix{T}, vProjFun :: Vector{<: Function}; numIterations :: N = 1000, δ :: T = 1e-6 ) where {T <: AbstractFloat, N <: Integer}
    # Projects mU[:, end] on the intersection of vProjFun
    # The initialization is assumed to be on mU[:, end].

    numSets = size(mU, 2);

    for kk ∈ 1:numIterations
        mU[:, 1] .= vProjFun[1](mU[:, end] .+ mZ[:, 1]);
        mZ[:, 1] .= mU[:, end] .+ mZ[:, 1] .- mU[:, 1];
        for ii ∈ 2:numSets
            mU[:, ii] .= vProjFun[ii](mU[:, ii - 1] .+ mZ[:, ii]);
            mZ[:, ii] .= mU[:, ii - 1] .+ mZ[:, ii] .- mU[:, ii];
        end
        if (maximum(abs.(mU[:, end] - mU[:, 1])) < δ)
            break;
        end 
    end

    return mU[:, numSets];

end

function PD3O!(mX :: Matrix{T}, vS :: Vector{T}, vX̄ :: Vector{T}, vT :: Vector{T}, hA :: Function, hAᵀ :: Function, h∇f :: Function, hProxG :: Function, hProxH :: Function, γ :: T, λ :: T) where {T <: AbstractFloat}

    numIterations = size(mX, 2);
    δ = λ / γ;
    μ = γ / λ; #<! (1 / δ)
    
    vX = @view mX[:, 1];
    
    for ii ∈ 2:numIterations
        vT .= vX - γ * h∇f(vX); #<! Buffer (vXH)
        
        vS .= vS + δ * hA(vX̄);
        vS .= vS - δ * hProxH(μ * vS, μ);
        vX = @view mX[:, ii];
        vX .= hProxG(vT - γ * hAᵀ(vS), γ);
        vX̄ .= 2vX - vT - γ * h∇f(vX);
    end

end


## Parameters

# Data
numRows = 5;
numCols = numRows; #<! Symmetric Matrix
δ       = 1e6;
δTol    = 1e-5;

# Solvers
numIterations = 250;

# Projected Gradient Descent
η = 0.005;

# PD3O

## Generate / Load Data
oRng = StableRNG(1234);
mB = randn(oRng, numRows, numCols);
# Will make `mB` SPSD to abs() in the DCP will make sense (https://math.stackexchange.com/questions/888677).
mB = mB' * mB + 0.1I;
mB = mB + mB';

# See https://discourse.julialang.org/t/73206
hδFun( mX :: Matrix{<: AbstractFloat} ) = δ * !(isapprox(tr(mX), 1; atol = δTol) && isapprox(mX, mX'; atol = δTol) && (eigmin(mX) > -δTol));
hObjFun( mX :: Matrix{<: AbstractFloat} ) = tr(mX * mB);# + hδFun(mX); #<! tr(mA * mB) = sum(mA' .* mB);

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
# Using Dykstra for projection onto intersection of Convex sets.
methodName = "PGD";

hVecToMat( vY :: Vector{T} ) where {T <: AbstractFloat} = reshape(vY, numRows, numCols);
vProjFun = [vec ∘ ProjSymMatSet ∘ hVecToMat, vec ∘ ProjPsdMatSet ∘ hVecToMat, vec ∘ ProjUniTrMatSet ∘ hVecToMat];
numSets = length(vProjFun);
mU = zeros(numRows * numCols, numSets);
mZ = zeros(numRows * numCols, numSets);

∇F( vY :: Vector{T} ) where {T <: AbstractFloat} = vec(mB');
hProjFun( vY :: Vector{T} ) where {T <: AbstractFloat} = ProjSetConDykstra(hUpMat!(mU, vY, numSets), mZ, vProjFun);

mX = zeros(numRows * numCols, numIterations);

for ii ∈ 2:numIterations
    mX[:, ii] = mX[:, ii - 1] .- η * ∇F(mX[:, ii - 1]); #<! Gradient step
    mX[:, ii] = hProjFun(mX[:, ii]); #<! Projection step
end

dSolvers[methodName] = [hObjFun(reshape(mX[:, ii], numRows, numCols)) for ii ∈ 1:size(mX, 2)];


# PD3O
# Solves: \arg \min_x f(x) + g(x) + h(A * x)
# f(x) = B * X 
# g(x) = δ_S_+(x) -> Prox_g(y) = Proj_S_+(y)
# h(x) = δ_Tr(x) ∈ [a, b] -> Prox_h(y) = clamp(y, a, b)
# Useful as it doesn't require solving big linear equation.
# γ - Primal step size, δ - Dual step size.

methodName = "PD3O";

fill!(mX, zero(eltype(mX)));

valL = 1; #<! Since h(A * x) = h(I * x) = h(x)
β = 1 / opnorm(mB); #<! By f(x)
γ = 1.8β; #<! γ < 2β (Like a primal step size for Gradient Descent)
λ = 250β * β; #<! γ * δ < (1 / (L * L))
μ = γ / λ; #<! In the paper 1/δ
δ = λ / γ;

hA( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = vX;
hAᵀ( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = vX;
h∇f( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = vec(mB');
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vec(ProjSPSDMatSet!(reshape(vY, numRows, numCols)));
hProxH( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = vec(ProjUniTrMatSet!(reshape(vY, numRows, numCols)));

vX̄ = copy(mX[:, 1]);
vS = copy(mX[:, 1]);
vT = zeros(numRows * numCols);

PD3O!(mX, vS, vX̄, vT, hA, hAᵀ, h∇f, hProxG, hProxH, γ, λ);

dSolvers[methodName] = [hObjFun(reshape(mX[:, ii], numRows, numCols)) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function, Condition Number = $(@sprintf("%0.3f", cond(mB)))", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end