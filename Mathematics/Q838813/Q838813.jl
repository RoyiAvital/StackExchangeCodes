# StackExchange Mathematics Q838813
# https://math.stackexchange.com/questions/838813
# Projection onto Birkhoff Polytope (Doubly Stochastic Matrices).
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  The paper Learning a Bi Stochastic Data Similarity Matrix enforces symmetry (As a similarity matrix).
#   3.  The paper merges the projection into Symmetric Matrix and Row Sum of 1 per row, into a single operation. 
#       Yet it seems that speed wise, it is better to apply twice.
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

function CVXSolver( mY :: AbstractMatrix{T} )  where {T <: AbstractFloat}
    
    numRows = size(mY, 1);

    mX = Variable(numRows, numCols);
    sConvProb = minimize( Convex.sumsquares(mX - mY), [Convex.sum(mX, dims = 2) == one(T), mX >= zero(T), mX == mX'] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return mX.value;

end

## Parameters

# Problem parameters
numRows = 200; #<! Matrix A
numCols = numRows;  #<! Matrix A


# Solver Parameters
numIterations   = Unsigned(50_000);

#%% Load / Generate Data
mY = rand(oRng, numRows, numCols);


## Analysis

# Using DCP Solver (Convex.jl)
mXRef = CVXSolver(mY);

# Using POCS
hProjNonNeg( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}   = max.(mY, zero(T));
hProjUnitCol( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}  = mY .- ((sum(mY, dims = 1) .- one(T)) ./ size(mY, 1));
hProjUnitRow( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}  = mY .- ((sum(mY, dims = 2) .- one(T)) ./ size(mY, 2));
hProjSym( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}      = (mY .+ mY') ./ T(2);

vProjFun = [hProjNonNeg, hProjUnitCol, hProjUnitRow, hProjSym];

mX = OrthogonalProjectionOntoConvexSets(mY, vProjFun);


## Display Results

resAnalysis = @sprintf("The maximum absolute deivation between the reference solution and the numerical solution is: %0.5f", sum(abs2, mX - mXRef));
println(resAnalysis);
resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", sum(abs2, mXRef - mY));
println(resAnalysis);
resAnalysis = @sprintf("The numerical solution optimal value is: %0.5f", sum(abs2, mX - mY));
println(resAnalysis);

# Run Time Analysis
mYY = rand(oRng, numRows, numCols);

runTime = @belapsed CVXSolver(mYY) seconds = 2;
resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed OrthogonalProjectionOntoConvexSets(mZ, $vProjFun) setup = (mZ = copy(mYY)) evals = 1 seconds = 2;
resAnalysis = @sprintf("The numerical solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);





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
