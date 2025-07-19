# StackExchange Mathematics Q3964387
# https://math.stackexchange.com/questions/3964387
# Solve ${\left\| \boldsymbol{A} \boldsymbol{x} - \boldsymbol{b} \right\|}_{4}$ Using QCQP Formulation.
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
# - 1.0.000     19/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function CVXSolver( mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    numRows = size(mA, 1);
    numCols = size(mA, 2);

    vX = Variable(numCols);
    
    sConvProb = minimize( Convex.norm(mA * vX - vB, 4) ); #<! https://github.com/jump-dev/Convex.jl/issues/722
    Convex.solve!(sConvProb, SCS.Optimizer; silent = true);
    
    return vec(vX.value);

end


## Parameters

# Data
numRows = 50;
numCols = 20;

# Model
modelNorm = 4.0; #<! 1 <= modelNorm <= inf

# Solver Parameters
numIterations = Unsigned(15);


## Load / Generate Data

mA  = randn(oRng, numRows, numCols);
vXᶲ = randn(oRng, numCols); #<! Reference
vB  = mA * vXᶲ;

hObjFun( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = norm(mA * vX - vB, modelNorm);

dSolvers = Dict();


## Analysis
# The Model: \arg \minₓ || A x - b ||₄

# DCP Solver
methodName = "Convex.jl"

vXRef = CVXSolver(mA, vB);

dSolvers[methodName] = hObjFun(vXRef) * ones(numIterations);
optVal = hObjFun(vXRef);

# Iterative Reweighted Least Squares (IRLS)
methodName = "IRLS";

mX = zeros(numCols, numIterations);
vT  = zeros(numCols);
vW  = zeros(numRows);
mWA = zeros(size(mA));
mC  = zeros(numCols, numCols);
sBkWorkSpace = BunchKaufmanWs(mC);

for ii = 2:numIterations
    vZ = mX[:, ii - 1];
    vZ = IRLS!(vZ, mA, vB, vW, mWA, mC, vT, sBkWorkSpace; normP = modelNorm, numItr = UInt32(1));
    mX[:, ii] .= vZ;
end

# vX = IRLS(mA, vB; normP = modelNorm, numItr = numIterations);

dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function (Relative Error [dB])", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
end
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Objective Value");

hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

# Run Time Analysis
runTime = @belapsed CVXSolver(mA, vB) seconds = 2;
resAnalysis = @sprintf("The Convex.jl (SCS) solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

runTime = @belapsed IRLS(mA, vB; normP = modelNorm, numItr = numIterations) seconds = 2;
resAnalysis = @sprintf("The IRLS solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

