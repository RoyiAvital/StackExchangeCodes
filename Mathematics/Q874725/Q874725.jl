# StackExchange Mathematics Q874725
# https://math.stackexchange.com/questions/874725
# Solving L1 Regularized Linear SVM.
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
# - 1.0.000     27/09/2025  Royi Avital
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
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( vW :: Vector{T}, valB :: T, mX :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # λ * || w ||_1 + 0.5 * sum_i max( 0, 1 - y_i * (wᵀ x_i + b) )

    regLoss   = sum(abs, vW);
    hingeLoss = zero(T);

    for ii in 1:numSamples
        valS       = vY[ii] * (dot(view(mX, ii, :), vW) + valB);
        hingeLoss += max(zero(T), one(T) - valS);
    end

    return regLoss + hingeLoss;
    
end

function SolveCVX( mX :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # λ * || w ||_1 + 0.5 * sum_i max( 0, 1 - y_i * (wᵀ x_i + b) )

    # `mX` the samples (Row) matrix
    dataDim = size(mX, 2);

    vW     = Convex.Variable(dataDim);
    paramB = Convex.Variable(1);

    hingeLoss = Convex.sum(Convex.pos(T(1) - vY .* (mX * vW + paramB)));

    sConvProb = minimize( λ * Convex.norm_1(vW) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vW.value), paramB.value;
    
end

function ChamPock!( mX :: Matrix{T}, mK :: Matrix{T}, vY :: Vector{T}, vX̄ :: Vector{T}, hProxF⁺ :: Function, hProxG :: Function, σ :: T, τ :: T; θ :: T = T(1), useAccel :: Bool = false ) where {T <: AbstractFloat}
    # Solving using Chambolle Pock algorithm (Also called Primal Dual Hybrid Gradient (PDHG) Method).
    # Solves: \arg \min_x f(K x) + g(x), f: Y ➡ [0, inf), g: X ➡ [0, inf).
    # Assumes efficient ProxF⁺ and ProxG.
    # Following the notations of Wikipedia.
    # Accelerated variation with γ = θ.

    numIterations = size(mX, 2);

    τᵢ = τ;
    σᵢ = σ;
    θᵢ = θ;
    
    for ii ∈ 2:numIterations
        vT = view(mX, :, ii - 1); #<! Previous iteration
        vX = view(mX, :, ii);
        
        # Calculation of `vY` depends on f() and should be adapted per function
        vY .= hProxF⁺(vY + σᵢ * (mK * vX̄), σᵢ);
        vX .= hProxG(vT - (τᵢ * mK' * vY), τᵢ);

        if useAccel
            θᵢ = inv(sqrt(one(T) + T(2) * θ * τᵢ));
            τᵢ = θᵢ * τᵢ;
            σᵢ = σᵢ / θᵢ;
        end
        
        @. vX̄ = vX + (θᵢ * (vX - vT));
    end

end

function ADMM!(mX :: AbstractMatrix{T}, vZ :: AbstractVector{T}, vU :: AbstractVector{T}, mA :: AbstractMatrix{T}, hProxF :: Function, hProxG :: Function; ρ :: T = T(2.5), λ :: T = one(T)) where {T <: AbstractFloat}
    # Solves f(x) + λ g(Ax)
    # Where z = Ax, and g(z) has a well defined Prox.
    # ADMM for the case Ax + z = 0
    # ProxF(y) = \arg \minₓ 0.5ρ * || A x - y ||_2^2 + f(x) where y = z - u
    # ProxG(y) = \arg \minₓ 0.5ρ * || x - y ||_2^2 + λ g(x) where y = A x + u
    # Initialization by mX[:, 1]
    # Supports in place ProxG

    numIterations = size(mX, 2);
    
    for ii ∈ 2:numIterations
        vX = view(mX, :, ii);
        vZ .-= vU;
        vX .= hProxF(vZ, ρ);
        mul!(vZ, mA, vX);
        vZ .+= vU;
        vZ .= hProxG(vZ, λ / ρ);
        # vX .= hProxF(vZ - vU, ρ);
        # vZ .= hProxG(mA * vX + vU, λ / ρ);
        # vU  = vU + mA * vX - vZ
        vU .= mul!(vU, mA, vX, one(T), one(T)) .- vZ;
    end

    return mX;

end

function SolveLsL1Cd( mA :: AbstractMatrix{T}, vB :: AbstractVector{T}, λ :: T; numIterations :: N = 10 * max(size(mA)...), ϵ :: T = 1e-7 ) where {T <: AbstractFloat, N <: Integer}
    # Solves: 0.5 * ‖A * x - b‖² + λ * ‖x‖₁
    # Seems to be one of the fastest ways to solve the problem.

    # Precompute squared column norms
    vANorm      = sum(abs2, mA; dims = 1); #<! Vector of length n
    numElements = size(mA, 2);

    # Initialize by LS solution
    vX  = mA \ vB;
    vX1 = copy(vX); #<! Previous iteration

    for ii = 1:numIterations
        copy!(vX1, vX);
        for jj = 1:numElements
            vA = view(mA, :, jj);
            colNormSqr = vANorm[jj];

            vR = vB - (mA * vX) + vA * vX[jj];
            β  = dot(vA, vR);

            vX[jj] = sign(β) * max(zero(T), abs(β) - λ) / colNormSqr; #<! 1D Soft Threshold
        end
        
        # Test convergence
        vX1 .-= vX;
        if maximum(abs, vX1) < ϵ
            break;
        end
    end

    return vX;

end

function SolveLsTvChambolle( mA :: AbstractMatrix{T}, vB :: AbstractVector{T}, mD :: AbstractMatrix{T}, λ :: T, τ :: T, σ :: T; θ :: T = one(T), numIterations :: N = 10_000, ϵ :: T = 1e-7, useAccel :: Bool = false ) where {T <: AbstractFloat, N <: Integer}
    # Solves: 0.5 * ‖A * x - b‖² + λ * ‖D * x‖₁
    # Using Chambolle Pock method
    # ProxF(y) = \arg \min_x  0.5 * ​∥ A x − b ∥_2^2 ​+ 0.5 * (1 / τ) * ​∥ x − z∥_2^2
    # ProxG(y) = ProjL∞Ball_λ(y) = y ./ max(1, abs(y) / λ)

    # Initialize by LS solution
    vX = mA \ vB;
    vX̄ = copy(vX);
    vZ = copy(vX); #<! Buffer
    vX1 = copy(vX); #<! Previous iteration
    vY = zeros(T, size(mD, 1));
    vȲ = copy(vY); #<! Buffer

    # Solving the System
    sCholA = cholesky(mA' * mA + inv(τ) * I);
    vAb = mA' * vB;

    τᵢ = τ;
    σᵢ = σ;
    θᵢ = θ;

    for ii in 1:numIterations
        copy!(vX1, vX);
        
        # Dual Update
        # vY .+= σᵢ .* mD * vX̄;
        mul!(vȲ, mD, vX̄);
        vY .+= σᵢ .* vȲ;
        vY ./= max.(one(T), abs.(vY) ./ λ);

        # Primal Update
        # vZ = vX .- τᵢ .* (mD' * vY);
        # vX = sCholA \ (vAb + inv(τᵢ) * vZ);
        mul!(vZ, mD', vY);
        vZ .= vAb .+ inv(τᵢ) .* (vX .- τᵢ .* vZ);
        ldiv!(vX, sCholA, vZ);

        if useAccel
            # Does not work!
            # θᵢ = inv(sqrt(one(T) + T(2) * θ * τᵢ));
            θᵢ = T(1) / sqrt(one(T) + T(2) * θ * τᵢ);
            τᵢ = θᵢ * τᵢ;
            σᵢ = σᵢ / θᵢ;
        end

        # Extrapolation
        @. vX̄ = vX + θᵢ * (vX - vX1);

        # Test convergence
        vX1 .-= vX;
        if maximum(abs, vX1) < ϵ
            break;
        end
    end

    return vX;
end

function SolveLsL1CVX( mA :: AbstractMatrix{T}, vB :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat}
    # Solves: 0.5 * ‖A * x - b‖² + λ * ‖x‖₁
    
    dataDim = size(mA, 2);

    vX = Convex.Variable(dataDim);

    sConvProb = minimize( T(0.5) * Convex.sumsquares(mA * vX - vB) + λ * Convex.norm_1(vX) ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);

end

function SolveLsTvCVX( mA :: AbstractMatrix{T}, vB :: AbstractVector{T}, mD :: AbstractMatrix{T}, λ :: T ) where {T <: AbstractFloat}
    # Solves: 0.5 * ‖A * x - b‖² + λ * ‖D * x‖₁
    
    dataDim = size(mA, 2);

    vX = Convex.Variable(dataDim);

    sConvProb = minimize( T(0.5) * Convex.sumsquares(mA * vX - vB) + λ * Convex.norm_1(mD * vX) ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);

end

## Parameters

# Data


# SVM Model
λ = 0.1;

# Solvers
numIterations = 25_000;
θ = 0.2;
ρ = 0.35;


## Load / Generate Data

# From SK Learn Example (https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)
mX = [
     0.4 -0.7;
    -1.5 -1.0;
    -1.4 -0.9;
    -1.3 -1.2;
    -1.1 -0.2;
    -1.2 -0.4;
    -0.5  1.2;
    -1.5  2.1;
     1.0  1.0;
     1.3  0.8;
     1.2  0.5;
     0.2 -2.0;
     0.5 -2.4;
     0.2 -2.3;
     0.0 -2.7;
     1.3  2.1;
];

vY = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

numSamples = length(vY);
dataDim = size(mX, 2);

mW = zeros(dataDim + 1, numIterations);

dSolvers = Dict();


## Analysis

hObjFun(vX :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vX[1:dataDim], vX[end], mX, vY, λ);

# DCP Solver
# Solve λ * || w ||_1 + 0.5 * sum_i max( 0, 1 - y_i * (wᵀ x_i + b) )
    # y_i ∈ {-1, 1}
vW, paramB = SolveCVX(mX, vY, λ);

vWRef  = [vW; paramB];
optVal = hObjFun(vWRef);

# Primal Dual Hybrid Gradient (PDHG) Method
# Solves: \arg \min f(Kx) + g(x) : HingeLoss(A x) + λ || x ||_1
methodName = "PDHG";

mA = diagm(vY) * [mX ones(numSamples)]; #<! For large cases should be defined as Linear Operator

hProxF(vY :: Vector{T}, σ :: T) where {T <: AbstractFloat} = ProxHingeLoss(vY, σ);
hProxF⁺(vY :: Vector{T}, σ :: T) where {T <: AbstractFloat} = vY - σ * ProxHingeLoss(vY ./ σ, inv(σ));
hProxG(vY :: Vector{T}, τ :: T) where {T <: AbstractFloat} = [ProxL1Norm(vY[1:dataDim], τ * λ); vY[end]]; 

mW .= 0.0;

# σ * τ * || A ||_2^2 ≤ 1
σ = 0.85 / sqrt(OpNormSquaredApprox(mA));
τ = σ;

vU = zeros(numSamples);
vW̄ = zeros(dataDim + 1);

ChamPock!(mW, mA, vU, vW̄, hProxF⁺, hProxG, σ, τ; θ = θ);

dSolvers[methodName] = [hObjFun(mW[:, ii]) for ii ∈ 1:size(mW, 2)];

# ADMM Method
# Solves: \arg \min f(x) + g(z) : λ || E x ||_1 + HingeLoss(z) subject to A x = z
# ProxF(y) = \arg \minₓ 0.5ρ * || A x - y ||_2^2 + f(x) -> \arg \minₓ 0.5ρ * || A x - y ||_2^2 + λ || E x ||_1
methodName = "ADMM";

mA  = diagm(vY) * [mX ones(numSamples)]; #<! For large cases should be defined as Linear Operator
mE = [Float64.(collect(I(dataDim))) zeros(dataDim)];

function ProjFCVX( mA :: Matrix{T}, vY :: Vector{T}, ρ :: T, λ :: T ) where {T <: AbstractFloat}
    # Solves a variant of the L1 Prox
    #  0.5 * ρ || A x - y ||_2^2 + λ * || x[1:(end - 1)] ||_1

    dataDim = size(mA, 2);

    vX = Convex.Variable(dataDim);

    sConvProb = minimize( (λ / ρ) * Convex.norm_1(vX[1:(dataDim - 1)]) + T(0.5) * Convex.sumsquares(mA * vX - vY) ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vX.value);
    
end

σ = 0.85 / sqrt(OpNormSquaredApprox(mE));
τ = σ;

hProxF(vY :: Vector{T}, ρ :: T) where {T <: AbstractFloat} = SolveLsTvChambolle(mA, vY, mE, λ / ρ, τ, σ); 
# hProxF(vY :: Vector{T}, τ :: T) where {T <: AbstractFloat} = ProjFCVX(mA, vY, τ, λ); #<! Reference
hProxG(vY :: Vector{T}, σ :: T) where {T <: AbstractFloat} = ProxHingeLoss(vY, σ);

mW .= 0.0;

vZ = zeros(numSamples);
vU = zeros(numSamples);

ADMM!(mW, vZ, vU, mA, hProxF, hProxG; ρ = ρ);

dSolvers[methodName] = [hObjFun(mW[:, ii]) for ii ∈ 1:size(mW, 2)];

# numRows = 12;
# numCols = 8;
# mA = randn(numRows, numCols);
# vB = randn(numRows);
# mD = randn(numRows - 1, numCols);
# λ = 5 * rand();
# vXRef = SolveLsTvCVX(mA, vB, mD, λ);
# # σ * τ * || A ||_2^2 ≤ 1
# σ = 0.85 / sqrt(OpNormSquaredApprox(mD));
# τ = σ;
# θ = 0.97 * minimum(eigvals(mA' * mA));
# vX = SolveLsTvChambolle(mA, vB, mD, λ, τ, σ; θ = 1.0, numIterations = 500_000, useAccel = false);
# maximum(abs.(vXRef - vX))


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", line = attr(width = 3.0),
               text = methodName, name = methodName);
end
sLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = Plot(vTr, sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

