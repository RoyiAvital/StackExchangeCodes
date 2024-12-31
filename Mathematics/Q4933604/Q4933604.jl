# StackExchange Mathematics Q4933604
# https://math.stackexchange.com/questions/4933604
# Algorithm to Solve a Sparse Recovery Problem by Permutation Matrix.
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
# - 1.0.000     25/06/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
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

function QuantizeP!( mP :: AbstractMatrix{T} ) where {T <: AbstractFloat}
    # Will fails if the first rows have the same values which becomes maximum

    vIdx = Vector{CartesianIndex{2}}(undef, size(mP, 2));
    
    for ii ∈ 1:size(mP, 2)
        maxIdx = argmax(mP);
        mP[maxIdx[1], :] .= zero(T); #<! Zero the row
        mP[:, maxIdx[2]] .= zero(T); #<! Zero the column
        vIdx[ii] = maxIdx;
    end

    # Assigning 1
    for ii ∈ 1:size(mP, 2)
        mP[vIdx[ii]] = one(T);
    end

    return mP;

end


## Parameters

# Problem parameters
numRows = 200; #<! Matrix Y
numCols = 100;  #<! Matrix Y
numVals = 20;


# Solver Parameters
numIterations   = Unsigned(50_000);
η               = 1e-5;

#%% Load / Generate Data
mH = rand(oRng, numRows, numCols);
mP = zeros(numCols, numVals);
vβ = randn(numVals);
vY = randn(numRows);


## Analysis

# Using POCS
hProjNonNeg( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat}  = max.(mY, zero(T));
hProjUnitCol( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat} = mY .- ((sum(mY, dims = 1) .- one(T)) ./ size(mY, 1)) #!< Add to each column for 1
hProjUnitRow( mY :: AbstractMatrix{T} ) where {T <: AbstractFloat} = mY .- (((sum(mY, dims = 2) .- one(T)) ./ size(mY, 2)) .* (sum(mY, dims = 2) .> one(T)));
hGradFun( mP :: AbstractMatrix{T} ) where {T <: AbstractFloat}     = mH' * (mH * mP * vβ - vY) * vβ'; #<! ObjFun: 0.5 * || H * P * β - y ||^2

hProj = hProjNonNeg ∘ hProjUnitRow ∘ hProjUnitCol ∘ hProjNonNeg;

GradientDescentAccelerated(mP, numIterations, η, hGradFun; ProjFun = hProj);

mO = copy(mP);

mO = QuantizeP!(mO);

# Quantize mP

# Run Time Analysis


## Display Results

# resAnalysis = @sprintf("The maximum absolute deviation between the reference solution and the numerical solution is: %0.5f", norm(mX - mXRef, 1));
# println(resAnalysis);
# resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", sum(abs2, mXRef - mY));
# println(resAnalysis);
# resAnalysis = @sprintf("The numerical solution optimal value is: %0.5f", sum(abs2, mX - mY));
# println(resAnalysis);
# runTime = @belapsed CVXSolver($mY);
# resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);
# runTime = @belapsed OrthogonalProjectionOntoConvexSets($mY, $vProjFun);
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
