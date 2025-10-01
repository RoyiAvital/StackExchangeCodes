# StackExchange Mathematics Q2607494
# https://math.stackexchange.com/questions/2607494
# Solving Large Linear System Originated from Matrix Equation with Kronecker Product Vectorization.
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
# - 1.0.000     29/09/2025  Royi Avital
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
using LinearOperators;
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

function ObjFun( vD :: Vector{T}, mP :: Matrix{T}, mB :: Matrix{T} ) where {T <: AbstractFloat}
    # 0.5 * || mB * Diag(vD) * mB^T - P ||_F^2

    return T(0.5) * sum(abs2, mB * Diagonal(vD) * mB' - mP);
    
end

function SolveCVX( mP :: Matrix{T}, mB :: Matrix{T} ) where {T <: AbstractFloat}
    # \arg \min_d 0.5 * || B * Diag(d) * B^T - P ||_F^2

    numCols = size(mB, 2);
    vD      = Convex.Variable(numCols);

    sConvProb = minimize( T(0.5) * Convex.sumsquares(mB * diagm(vD) * mB' - mP) ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vD.value);
    
end

function LSMR!( mX :: AbstractMatrix{T}, mA :: Union{LinearOperator{T}, AbstractVector{T}}, vB :: AbstractVector, vU :: AbstractVector{T}, vV :: AbstractVector{T}, vW :: AbstractVector{T}, vAtu :: AbstractVector{T}, vAv :: AbstractVector{T} ) where {T <: AbstractFloat}
    # size(mA) = m, n
    # `vU` - Length of `vB`
    # `vV` - Length of `vX`
    # `vW` - Length of `vX`
    # `vAtu` - Length of `vX`
    # `vAv` - Length of `vB`

    # Initialization
    numIterations = size(mX, 2);

    vX = view(mX, :, 1);
    # u = b - A*x (residual)
    mul!(vU, mA, vX); #<! u = A*x
    @. vU = vB - vU;  #<! u = b - A*x
    β = norm(vU);     #<! β = ||u||

    @. vU = vU / β; #<! Normalize u

    mul!(vV, mA', vU); #<! v = A'*u
    α = norm(vV); #<! α = ||v||
    @. vV = vV / α; #<! normalize v

    copyto!(vW, vV); #<! w = v (search direction)
    θ̅  = α; #<! θ̅ = α (initial variables)
    ρ̅  = α; #<! ρ̅ = α
    φ̅  = β; #<! φ̅ = β

    for kk in 2:numIterations
        # Bidiagonalization step
        mul!(vAv, mA, vV);     #<! Av = A*v
        @. vAv = vAv - α * vU; #<! Av = Av - α*u
        β = norm(vAv);         #<! β = ||Av||
        @. vU = vAv / β;       #<! u = Av / β

        mul!(vAtu, mA', vU);     #<! Atv = A'*u
        @. vAtu = vAtu - β * vV; #<! Atv = Atv - β*v
        α = norm(vAtu);          #<! α = ||Atv||
        @. vV = vAtu / α;        #<! v = Atv / α

        # Construct and apply rotation
        ρ = sqrt(ρ̅  ^ 2 + β ^ 2); #<! ρ = sqrt(ρ̅^2 + β^2)
        c = ρ̅ / ρ;                #<! cos
        s = β / ρ;                #<! sin
        θ = s * α;                #<! θ = s*α
        ρ̅ = -c * α;               #<! ρ̅ = -c*α
        φ = c * φ̅ ;               #<! φ = c*φ̅
        φ̅ = s * φ̅ ;               #<! φ̅ = s*φ̅

        # Update x and w
        vX = view(mX, :, kk);
        vX1 = view(mX, :, kk - 1); #<! Previous iteration
        @. vX = vX1 + (φ / ρ) * vW; #<! x update
        @. vW = vV - (θ / ρ) * vW; #<! w update
    end

    return mX;

end

function GradientDescentAccelerated( mX :: AbstractMatrix{T}, η :: T, ∇ObjFun :: Function; ProjFun :: Function = identity ) where {T <: AbstractFloat} #, F <: Function, G <: Function}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    vX = view(mX, :, 1);
    vW = Array{T, ndims(vX)}(undef, size(vX));
    vZ = copy(vX);

    ∇vZ = Array{T, ndims(vX)}(undef, size(vX));

    for ii ∈ 2:size(mX, 2)
        # FISTA (Nesterov) Accelerated
    
        ∇vZ = ∇ObjFun(vZ);
    
        vW .= view(mX, :, ii - 1); #<! Previous iteration
        vX  = view(mX, :, ii); 
        vX .= vZ .- (η .* ∇vZ);
        vX .= ProjFun(vX);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        vZ .= vX .+ (fistaStepSize .* (vX .- vW));
    end

    return mX;

end

function LinOpKronProduct!( vD :: AbstractVector{T}, mA :: AbstractMatrix{T},  mB :: AbstractMatrix{T}, vC :: AbstractVector{T}, mT :: AbstractMatrix{T}; isDiag :: Bool = false ) where {T <: AbstractFloat}
    # Implements vD = (mB' ⊗ mA) * vC
    # Equivalent of vD = vec(mA * mC * mB) where mC = mat(vC)
    # TODO: Optimize the order `(mA * mC) * mB` or `mA * (mC * mB)`

    numRowsA = size(mA, 1);
    numColsA = size(mA, 2);
    numRowsB = size(mB, 1);
    numColsB = size(mB, 2);

    mC = reshape(vC, numColsA, numRowsB);
    if isDiag
        mC = Diagonal(mC);
    end
    mD = reshape(vD, numColsB, numRowsA);
    
    mul!(mT, mA, mC);
    mul!(mD, mT, mB);

end

function LinOpKronProductT!( vD :: AbstractVector{T}, mA :: AbstractMatrix{T},  mB :: AbstractMatrix{T}, vC :: AbstractVector{T}, mT :: AbstractMatrix{T}; isDiag :: Bool = false ) where {T <: AbstractFloat}
    # Implements vD = (mB' ⊗ mA)' * vC -> vD = (mB ⊗ mA') * vC
    # Equivalent of vD = vec(mA' * mC * mB') where mC = mat(vC)
    # TODO: Optimize the order `(mA' * mC) * mB' or `mA' * (mC * mB')`

    numRowsA = size(mA, 1);
    numColsA = size(mA, 2);
    numRowsB = size(mB, 1);
    numColsB = size(mB, 2);

    mC = reshape(vC, numRowsA, numColsB);
    if isDiag
        mC = Diagonal(mC);
    end
    mD = reshape(vD, numRowsB, numColsA);
    
    mul!(mT, mA', mC);
    mul!(mD, mT, mB');

end

function LinOpKronProductDiag!( vO :: AbstractVector{T}, mA :: AbstractMatrix{T},  mB :: AbstractMatrix{T}, vD :: AbstractVector{T}, mT :: AbstractMatrix{T} ) where {T <: AbstractFloat}
    # Implements vO = (mB' ⊗ mA)[:, vI] * vD
    # Equivalent of vO = vec(mA * Diag(vD) * mB)
    # mA * diagm(vD) ⇄ mA .* vD'
    # diagm(vD) * mB ⇄ vD .* mB

    numRowsA = size(mA, 1);
    numColsA = size(mA, 2);
    numRowsB = size(mB, 1);
    numColsB = size(mB, 2);

    mO = reshape(vO, numColsB, numRowsA);
    
    # mul!(mT, mA, mC);
    mT .= mA .* vD';
    mul!(mO, mT, mB);

end

function LinOpKronProductDiagT!( vO :: AbstractVector{T}, mA :: AbstractMatrix{T},  mB :: AbstractMatrix{T}, vD :: AbstractVector{T}, vT :: AbstractVector{T} ) where {T <: AbstractFloat}
    # Implements vO = ((mB' ⊗ mA)[:, vI])' * vD
    # Equivalent of vO = diag(mA' * mat(vD) * mB')
    # size(vO) = (numColsA, )

    numRowsA = size(mA, 1);
    numColsA = size(mA, 2);
    numRowsB = size(mB, 1);
    numColsB = size(mB, 2);

    vI = diagind(numCols, numCols);

    for ii in 1:numColsA
        vT = GetIColKron!(vT, mB', mA, vI[ii]);
        vO[ii] = dot(vT, vD);
    end

end

function GetIColKron!( vO :: AbstractVector{T}, mA :: AbstractMatrix{T}, mB :: AbstractMatrix{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer}
    # Gets the `ii` column of `mA ⊗ mB`
    
    m, n = size(mA);
    p, q = size(mB);

    # Map ii → (jA, jB)
    jB = (ii - 1) % q + 1;
    jA = (ii - 1) ÷ q + 1;

    vA = @view mA[:, jA];
    vB = @view mB[:, jB];

    # Fill vO with vA ⊗ vB
    idx = 1;
    for a in vA
        for b in vB
            vO[idx] = a * b;
            idx += 1;
        end
    end

    return vO;

end

function GetIColKron( mA :: AbstractMatrix{T}, mB :: AbstractMatrix{T}, ii :: N ) where {T <: AbstractFloat, N <: Integer}
    # Gets the `ii` column of `mA ⊗ mB`

    numRowsA = size(mA, 1);
    numColsA = size(mA, 2);
    numRowsB = size(mB, 1);
    numColsB = size(mB, 2);

    vO = zeros(T, numRowsA * numRowsB);
    vO = GetIColKron!(vO, mA, mB, ii);

    return vO;

end


## Parameters

# Data
numRows = 576;
numCols = 1296;

numRows = 576 ÷ 4;
numCols = 1296 ÷ 4;

# Solvers
numIterations = 200;
η = 1e-5;


## Load / Generate Data

mB = randn(oRng, numRows, numCols);
mP = randn(oRng, numRows, numRows);

dSolvers = Dict();

mX = zeros(numCols, numIterations);


## Analysis

hObjFun(vD :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vD, mP, mB);

# DCP Solver

vDRef  = SolveCVX(mP, mB);
optVal = hObjFun(vDRef);

# Primal Dual Hybrid Gradient (PDHG) Method
# Solves: \arg \min f(Kx) + g(x) : HingeLoss(A x) + λ || x ||_1
methodName = "Accelerated Gradient Descent";

hGradF(vD :: AbstractVector{T}) where {T <: AbstractFloat} = diag(mB' * (mB * Diagonal(vD) * mB' - mP) * mB);

mX = GradientDescentAccelerated(mX, η, hGradF);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# LSMR Method
# Solves: `vP = (mB ⊗ mB) * vD` where `vP = vec(mP)`
methodName = "LSMR";

# mT1 = zeros(numRows, numCols);
# mT2 = zeros(numCols, numRows);

# mBB = LinearOperator(Float64, numRows * numRows, numCols * numCols, false, false, (vO, vI) -> LinOpKronProduct!(vO, mB, mB', vI, mT1; isDiag = true), (vO, vI) -> LinOpKronProductT!(vO, mB, mB', vI, mT2; isDiag = true)); #<! Stand for `mB ⊗ mB`


mT  = zeros(numRows, numCols);
vT  = zeros(numRows * numRows);
mBB = LinearOperator(Float64, numRows * numRows, numCols, false, false, (vO, vI) -> LinOpKronProductDiag!(vO, mB, mB', vI, mT), (vO, vI) -> LinOpKronProductDiagT!(vO, mB, mB', vI, vT)); #<! Stand for `(mB ⊗ mB)[:, vI]`

# Variables for `LSMR!()`
vX0 = zeros(size(mBB, 2));
vP  = vec(mP);

vU   = zero(vP);
vV   = zero(vX0);
vW   = zero(vX0);
vAtu = zero(vX0);
vAv  = zero(vP);

mX .= 0.0;
mX = LSMR!(mX, mBB, vP, vU, vV, vW, vAtu, vAv);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


# Check Run Time
mX = zeros(numCols, 10);
@btime GradientDescentAccelerated($mX, $η, $hGradF);
@btime LSMR!($mX, $mBB, $vP, $vU, $vV, $vW, $vAtu, $vAv);


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

