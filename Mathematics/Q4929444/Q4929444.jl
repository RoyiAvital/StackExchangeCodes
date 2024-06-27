# StackExchange Mathematics Q4929444
# https://math.stackexchange.com/questions/4938099
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
# using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = false;

oRng = StableRNG(1234);

## Functions

function CVXSolver( vY :: AbstractVector{T}, mK :: AbstractMatrix{T}, ε :: T, λ :: T )  where {T <: AbstractFloat}
    
    numRows = size(mK, 1);
    
    vα = Variable(numRows);
    sConvProb = minimize( 0.5 * Convex.quadform(vα, mK, assume_psd = true) - Convex.dot(vα, vY) + ε * Convex.norm(vα, 1), [abs(vα) <= (one(T) / (T(2.0) * numRows * λ))] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return vα.value;

end

## Parameters

# Problem parameters
numRows = 20; #<! Matrix K
numCols = numRows;  #<! Matrix K

ε = 0.5;
λ = 0.7;


# Solver Parameters
numIterations   = Unsigned(50_000);
η = 1e-4;

#%% Load / Generate Data
mK = rand(oRng, numRows, numCols);
mK = mK' * mK;
vY = rand(oRng, numRows);


## Analysis

# Using DCP Solver (Convex.jl)

vXRef = CVXSolver(vY, mK, ε, λ);

# Using Proximal Gradient Descent / Proximal Gradient Method (PGM)
∇F( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = mK * vX - vY;
hProxL1( vX :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat} = max.(vX .- λ, zero(T)) + min.(vX .- λ, zero(T));
hProxClamp( vX :: AbstractVector{T}, _ :: T ) where {T <: AbstractFloat} = clamp.(vX, -one(T) / (T(2) * length(vX) * λ), one(T) / (T(2) * length(vX) * λ));

hProxFun( vX :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat} = hProxClamp(hProxL1(vX, λ), λ);

vX = zeros(numRows);

vX = ProximalGradientDescent(vX, ∇F, hProxFun, η, 100_000; λ = ε);



## Display Results

resAnalysis = @sprintf("The maximum absolute deivation between the reference solution and the numerical solution is: %0.5f", norm(vX - vXRef, 1));
println(resAnalysis);
resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", sum(abs2, vXRef - vY));
println(resAnalysis);
resAnalysis = @sprintf("The numerical solution optimal value is: %0.5f", sum(abs2, vX - vY));
println(resAnalysis);

# Run Time Analysis
# vYY = rand(oRng, numCols);

# runTime = @belapsed CVXSolver($vY, mA, vB);
# resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);
# runTime = @belapsed ProximalGradientDescent(vZ, ∇F, hProxFun, η, numIterations; λ = λ) setup = (vZ = copy(vYY));
# resAnalysis = @sprintf("The numerical solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);





# figureIdx += 1;

# titleStr = L"\\alpha_{1c} = 352 \pm 11 \\text{ km s}^{-1}";

# oTrace1 = scatter(x = 1:numIterations, y = vObjFun, mode = "lines", text = "Gradient Descent", name = "Gradient Descent",
#                   line = attr(width = 3.0));
# oTrace2 = scatter(x = 1:numIterations, y = optVal * ones(numIterations), 
#                   mode = "lines", text = "Optimal Value", name = "Optimum (Convex.jl)",
#                   line = attr(width = 1.5, dash = "dot"));
# oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = "Value");
# hP = plot([oTrace1, oTrace2], oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end
