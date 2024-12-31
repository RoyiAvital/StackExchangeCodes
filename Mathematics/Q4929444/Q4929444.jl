# StackExchange Mathematics Q4929444
# https://math.stackexchange.com/questions/4929444
# Solve the Soft SVM Dual Problem with L1 Regularization.
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
# - 1.0.000     27/06/2024  Royi Avital
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
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function CVXSolver( vY :: AbstractVector{T}, mK :: AbstractMatrix{T}, ε :: T, λ :: T )  where {T <: AbstractFloat}
    
    numRows = size(mK, 1);
    
    vα = Variable(numRows);
    sConvProb = minimize( 0.5 * Convex.quadform(vα, mK, assume_psd = true) - Convex.dot(vα, vY) + ε * Convex.norm(vα, 1), [abs(vα) <= (one(T) / (T(2.0) * numRows * λ))] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return vec(vα.value);

end

function DispSolverSummary( vX, vXRef, hObjFun, methodName )
    resAnalysis = @sprintf("The maximum absolute deviation between the reference solution and the %s solution is: %0.5f", methodName, norm(vX - vXRef, 1));
    println(resAnalysis);
    resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", hObjFun(vXRef));
    println(resAnalysis);
    resAnalysis = @sprintf("The %s solution optimal value is: %0.5f", methodName, hObjFun(vX));
    println(resAnalysis);
end

## Parameters

# Problem parameters
numRows = 500; #<! Matrix K
numCols = numRows;  #<! Matrix K

ε = 0.5;
λ = 0.7;


# Solver Parameters
numIterations   = Unsigned(7_500);
η               = 1e-6;
ρ               = 50.0;

#%% Load / Generate Data
mK = rand(oRng, numRows, numCols);
mK = mK' * mK;
vY = rand(oRng, numRows);

hObjFun( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = 0.5 * dot(vX, mK, vX) - dot(vX, vY) + ε * norm(vX, 1);

dSolvers = Dict();


## Analysis
# The model: f(x) + ε g(x)
# f(x) = 0.5 * x' K x - x' y
# g(x) = ||x||_1
# ∇f(x) = K * x - y
# ProxG(y) = SoftThr(y, ε)
# ProxF(z) = \arg \minₓ (ρ / 2) * || x - z ||_2^2 + f(x)


# Using DCP Solver (Convex.jl)

methodName = "Convex.jl";

vXRef = CVXSolver(vY, mK, ε, λ);
optVal = hObjFun(vXRef);
dSolvers[methodName] = optVal * ones(numIterations);

# Using Proximal Gradient Descent / Proximal Gradient Method (PGM)
methodName = "Accelerated PGM";

∇F( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = mK * vX - vY;
hProxL1( vX :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat} = max.(vX .- λ, zero(T)) .+ min.(vX .+ λ, zero(T));
hProxClamp( vX :: AbstractVector{T}, _ :: T ) where {T <: AbstractFloat} = clamp.(vX, -one(T) / (T(2) * length(vX) * λ), one(T) / (T(2) * length(vX) * λ));

# Composing the projection onto the constraint onto the ProxG.
hProxFun( vX :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat} = hProxClamp(hProxL1(vX, λ), λ);

mX = zeros(numCols, numIterations);
vG = zeros(numCols);
vZ = zeros(numCols);
vW = zeros(numCols);
for ii = 2:numIterations
    vX = view(mX, :, ii);
    vX .= mX[:, ii - 1];
    vX = ProximalGradientDescentAcc!(vX, vG, vZ, vW, ∇F, hProxFun, η, 1; λ = ε);
end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# vX = ProximalGradientDescentAcc(zeros(numRows), ∇F, hProxFun, η / 1000, numIterations; λ = ε);
# DispSolverSummary(vX, vXRef, hObjFun, methodName);

DispSolverSummary(mX[:, end], vXRef, hObjFun, methodName);

# Using ADMM
methodName = "ADMM";
oCholK = cholesky(mK + ρ * I);
hProxF( vZ :: AbstractVector{T}, ρ :: T ) where {T <: AbstractFloat} = oCholK \ (ρ * vZ + vY);

mA = 1.0I(numRows);

mX = zeros(size(mA, 2), numIterations);
vZ = hProxFun(mA * mX[:, 1] + vU, ε / ρ);
vU = mA * mX[:, 1] - vZ;
for ii = 2:numIterations
    vX = view(mX, :, ii);
    vX .= mX[:, ii - 1];
    vX = ADMM!(vX, vZ, vU, mA, hProxF, hProxFun, 1; ρ = ρ, λ = ε);
end

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

DispSolverSummary(mX[:, end], vXRef, hObjFun, methodName);



## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0));
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0));
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Run Time Analysis
vYY = rand(oRng, numCols);

runTime = @belapsed CVXSolver($vYY, mK, ε, λ)  seconds = 2;
resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed ProximalGradientDescentAcc(vZ, ∇F, hProxFun, η, numIterations; λ = λ) setup = (vZ = copy(vYY)) seconds = 2;
resAnalysis = @sprintf("The PGM solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed ADMM(vZ, mA, hProxF, hProxFun, 300; ρ = ρ, λ = ε) setup = (vZ = copy(vYY)) seconds = 2;
resAnalysis = @sprintf("The ADMM solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
