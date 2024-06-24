# StackExchange Mathematics Q4936790
# https://math.stackexchange.com/questions/4936790
# Orthogonal Projection onto the Convex Hull of Permutation Matrix.
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
# - 1.0.000     24/06/2024  Royi Avital
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

exportFigures = false;

oRng = StableRNG(1234);

## Functions

function ValidReciprocal!( vA :: AbstractVecOrMat{T}, vB :: AbstractVecOrMat{T} )  where {T <: AbstractFloat}
    # Calculates the multiplicative inverse (Reciprocal) if it is valid (Not zero).
    
    for ii ∈ eachindex(vB)
        vA[ii] = ifelse(vB[ii] != zero(T), one(T) / vB[ii], zero(T));
    end

end

function SinkhornScaling!( mA :: Matrix{T}; numIterations :: S = 1_000, ε :: T = T(1e-8) ) where {T <: AbstractFloat, S <: Integer}
    # Converges for:
    # 1. The input is a square matrix.
    # 1. Elements are non negative.
    # 2. Sum of each row / column is positive.

    # vC = one(T) ./ sum(mA, dims = 1);
    # vR = one(T) ./ (mA * vC');
    
    vC = zeros(T, (1, size(mA, 2)));
    vR = zeros(T, (size(mA, 1), 1));

    ValidReciprocal!(vC, sum(mA, dims = 1));
    ValidReciprocal!(vR, mA * vC');

    for ii ∈ 1:numIterations
        vCInv = (vR' * mA) .* 5;
        (maximum(abs.((vCInv .* vC) .- 1)) <= ε) && break;
        # vC .= one(T) ./ vCInv;
        # vR = one(T) ./ (mA * vC');
        ValidReciprocal!(vC, vCInv);
        ValidReciprocal!(vR, mA * vC');
    end

    mA .*= vR * vC;

end

function CVXSolver( mY :: AbstractArray{T} ) where {T <: AbstractFloat}

    mX = Variable(numRows, numCols);
    sConvProb = minimize( Convex.sumsquares(mX - mY), [Convex.sum(mX, dims = 1) == 1.0, Convex.sum(mX, dims = 2) <= 1.0, mX >= 0.0] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return mX.value;
    
end

## Parameters

# Problem parameters
numRows = 200; #<! Matrix Y
numCols = 100;  #<! Matrix Y


# Solver Parameters
numIterations   = Unsigned(50_000);

#%% Load / Generate Data
mY = rand(oRng, numRows, numCols);


## Analysis

# Using DCP Solver (Convex.jl)
mX = Variable(numRows, numCols);
sConvProb = minimize( Convex.sumsquares(mX - mY), [Convex.sum(mX, dims = 1) == 1.0, Convex.sum(mX, dims = 2) <= 1.0, mX >= 0.0] );
solve!(sConvProb, ECOS.Optimizer; silent = true);
mXRef = mX.value;
optVal = sConvProb.optval;

# Using POCS
hProjNonNeg( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}  = max.(mY, zero(T));
hProjUnitCol( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat} = mY .- ((sum(mY, dims = 1) .- one(T)) ./ size(mY, 1)) #!< Add to each column for 1
hProjUnitRow( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat} = mY .- (((sum(mY, dims = 2) .- one(T)) ./ size(mY, 2)) .* (sum(mY, dims = 2) .> one(T)));

vProjFun = [hProjNonNeg, hProjUnitCol, hProjUnitRow];

mX = OrthogonalProjectionOntoConvexSets(mY, vProjFun);

# Run Time Analysis


## Display Results

resAnalysis = @sprintf("The maximum absolute deivation between the reference solution and the numerical solution is: %0.5f", norm(mX - mXRef, 1));
println(resAnalysis);
resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", sum(abs2, mXRef - mY));
println(resAnalysis);
resAnalysis = @sprintf("The numerical solution optimal value is: %0.5f", sum(abs2, mX - mY));
println(resAnalysis);
runTime = @belapsed CVXSolver($mY);
resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed OrthogonalProjectionOntoConvexSets($mY, $vProjFun);
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
