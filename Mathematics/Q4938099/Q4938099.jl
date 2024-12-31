# StackExchange Mathematics Q4938099
# https://math.stackexchange.com/questions/4938099
# Orthogonal Projection onto a Polyhedron (Matrix Inequality).
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
# - 1.0.000     26/06/2024  Royi Avital
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

function CVXSolver( vY :: AbstractVector{T}, mA :: AbstractMatrix{T}, vB :: AbstractVector{T} )  where {T <: AbstractFloat}
    
    vX = Variable(numCols);
    sConvProb = minimize( Convex.sumsquares(vX - vY), [mA * vX <= vB] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return vX.value;

end

## Parameters

# Problem parameters
numRows = 200; #<! Matrix A
numCols = 100;  #<! Matrix A


# Solver Parameters
numIterations   = Unsigned(50_000);

#%% Load / Generate Data
mA = rand(oRng, numRows, numCols);
vB = rand(oRng, numRows);
vY = rand(oRng, numCols);


## Analysis

# Using DCP Solver (Convex.jl)
vX = Variable(numCols);
sConvProb = minimize( Convex.sumsquares(vX - vY), [mA * vX <= vB] );
solve!(sConvProb, ECOS.Optimizer; silent = true);
vXRef = vX.value;
optVal = sConvProb.optval;

# Using POCS
hProjHalfSpace( vY :: AbstractVector{T}, vA :: AbstractVector{T}, b :: T ) where {T <: AbstractFloat} = ifelse(vA' * vY > b, vY - ((vA' * vY - b) / sum(abs2, vA)) .* vA, vY);

# vProjFun = Vector{Function}(undef, numRows);
# for ii ∈ numRows
#     vProjFun[ii] = vY -> hProjHalfSpace(vY, mA[ii, :], vB[ii]);
# end
vProjFun = [vY -> hProjHalfSpace(vY, mA[ii, :], vB[ii]) for ii ∈ 1:numRows];

vX = OrthogonalProjectionOntoConvexSets(vY, vProjFun);



## Display Results

resAnalysis = @sprintf("The maximum absolute deviation between the reference solution and the numerical solution is: %0.5f", norm(vX - vXRef, 1));
println(resAnalysis);
resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", sum(abs2, vXRef - vY));
println(resAnalysis);
resAnalysis = @sprintf("The numerical solution optimal value is: %0.5f", sum(abs2, vX - vY));
println(resAnalysis);

# Run Time Analysis
vY = rand(oRng, numCols);

runTime = @belapsed CVXSolver($vY, mA, vB);
resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed OrthogonalProjectionOntoConvexSets(vZ, $vProjFun) setup = (vZ = copy(vY));
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
