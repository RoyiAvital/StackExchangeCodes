# StackExchange Mathematics Q4952808
# https://math.stackexchange.com/questions/4952808
# Solve argminB∥Avec(B)∥22 subject to rank(B)=2.
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
# - 1.0.000     31/07/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
# using BenchmarkTools;
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

function SchattenNorm( mA :: Matrix{T}, p :: T ) where {T <: AbstractFloat}

    return norm(svdvals(mA), p);
    
end

function SoftThr( vX :: Vector{T}, λ :: T ) where {T <: AbstractFloat}

    return sign.(vX) .* max.((abs.(vX) .- λ), zero(T));

end

function ProxF( vF :: Vector{T}, λ :: T ) where {T <: AbstractFloat}

    mF = reshape(vF, (3, 3)); #<! View
    oSvd = svd(mF);
    vS = SoftThr(oSvd.S, λ);
    
    mF = oSvd.U * Diagonal(vS) * oSvd.Vt; #<! Instance

    return mF[:];
    
end

## Parameters

# Problem parameters
numRows = 200; #<! Matrix A
numCols = 9;  #<! Matrix A


# Solver Parameters
numIterations   = Unsigned(50_000);

#%% Load / Generate Data
mA = rand(oRng, numRows, numCols);
λ  = 0.5;



oSvd = svd(mA);
mF   = reshape(oSvd.V[:, end], (3, 3));
vF   = vec(mF); #<! View

mFF = zeros(9, numGridPts);

numGridPts = 1000;
vλ = LinRange(0, 10, numGridPts);
mR = zeros(numGridPts, 2);

for ii ∈ 1:numGridPts
    vFF = ProxF(vF, vλ[ii]);
    mFF[:, ii] = vFF;
    mR[ii, 1] = rank(reshape(vFF, (3, 3)));
    mR[ii, 2] = sum(abs2, mA * vFF);
end















## Analysis

# Using DCP Solver (Convex.jl)
mB = Variable(3, 3);
sConvProb = minimize( Convex.sumsquares(mA * Convex.vec(mB)) + λ * Convex.nuclearnorm(mB) );
Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
# vXRef = vX.value;
# optVal = sConvProb.optval;

# Using POCS
hProjHalfSpace( vY :: AbstractVector{T}, vA :: AbstractVector{T}, b :: T ) where {T <: AbstractFloat} = ifelse(vA' * vY > b, vY - ((vA' * vY - b) / sum(abs2, vA)) .* vA, vY);

# vProjFun = Vector{Function}(undef, numRows);
# for ii ∈ numRows
#     vProjFun[ii] = vY -> hProjHalfSpace(vY, mA[ii, :], vB[ii]);
# end
vProjFun = [vY -> hProjHalfSpace(vY, mA[ii, :], vB[ii]) for ii ∈ 1:numRows];

vX = OrthogonalProjectionOntoConvexSets(vY, vProjFun);



## Display Results

resAnalysis = @sprintf("The maximum absolute deivation between the reference solution and the numerical solution is: %0.5f", norm(vX - vXRef, 1));
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
