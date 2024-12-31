# StackExchange Mathematics Q1275192
# https://math.stackexchange.com/questions/1275192
# Optimization of Sum of Logs of Affine Function.
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
# - 1.0.000     30/12/2024  Royi Avital
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
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function DispSolverSummary( vX :: VecOrMat{T}, vXRef :: VecOrMat{T}, hObjFun :: Function, methodName :: S ) where {T <: AbstractFloat, S <: AbstractString}
    resAnalysis = @sprintf("The maximum absolute deviation between the reference solution and the %s solution is: %0.5f", methodName, norm(vX - vXRef, 1));
    println(resAnalysis);
    resAnalysis = @sprintf("The reference solution optimal value is: %0.5f", hObjFun(vXRef));
    println(resAnalysis);
    resAnalysis = @sprintf("The %s solution optimal value is: %0.5f", methodName, hObjFun(vX));
    println(resAnalysis);
end

function CVXSolver( mK :: AbstractMatrix{T} ) where {T <: AbstractFloat}
    
    numRows = size(mK, 1);
    
    vX = Variable(numRows);
    sConvProb = minimize( sum(-log(one(T) + mK' * vX)), [norm(vX) <= one(T)] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    return vec(vX.value);

end

function f( vX :: Vector{T}, mK :: Matrix{T} ) where {T <: AbstractFloat}

    numCols = size(mK, 2);
    f       = zero(T);

    for ii ∈ 1:numCols
        f += -log1p(dot(view(mK, :, ii), vX));
    end

    return f;
    
end

function ∇f( vX :: Vector{T}, mK :: Matrix{T} ) where {T <: AbstractFloat}

    numRows = size(mK, 1);
    numCols = size(mK, 2);

    vG = zeros(T, numRows);

    for ii ∈ 1:numCols
        vG .+= (-one(T) / (one(T) + dot(view(mK, :, ii), vX))) .* view(mK, :, ii);
    end

    return vG;

end


## Parameters

numRows = 250; #<! Number of elements
numCols = 500; #<! Number of equations
ϵ       = 1e-5;

# Solver
numIter = 250;
η       = 1e-3;

# Visualization
numGridPts = 500;


## Load / Generate Data

# Keep columns of mK on Unit Ball
mK = randn(oRng, numRows, numCols);
mK = mK ./ reshape(norm.(eachcol(mK)), (1, numCols));

dSolvers = Dict();


## Analysis

hF( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = f(vX, mK); #<! Objective function
h∇f( vX :: AbstractVector{T} ) where {T <: AbstractFloat} = ∇f(vX, mK); #<! Gradient function
hProjFun( vY :: AbstractVector{T}, λ :: T ) where {T <: AbstractFloat} = ProjectL2Ball(vY; ballRadius = 1.0); #<! Projection function

# Validate Gradient
vX = randn(oRng, numRows);
vX = normalize(vX);
@assert (maximum(abs.(h∇f(vX) - CalcFunGrad(vX, hF))) <= ϵ) "The gradient calculation is not verified";

# Using DCP Solver (Convex.jl)
methodName = "Convex.jl";

vXRef = CVXSolver(mK);
optVal = hF(vXRef);
dSolvers[methodName] = optVal * ones(numIter);

# Using Proximal Gradient Descent / Proximal Gradient Method (PGM)
methodName = "Accelerated PGM";

mX = zeros(numRows, numIter);
vG = zeros(numRows);
vZ = zeros(numRows);
vW = zeros(numRows);
for ii = 2:numIter
    vX = view(mX, :, ii);
    vX .= mX[:, ii - 1]; #<! Update previous iteration as current
    vX = ProximalGradientDescentAcc!(vX, vG, vZ, vW, h∇f, hProjFun, η, 1);
end

dSolvers[methodName] = [hF(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

DispSolverSummary(mX[:, end], vXRef, hF, methodName);


## Display Results

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = 1:numIter, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
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
    vTr[ii] = scatter(x = 1:numIter, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
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
vZZ = zeros(numRows);

runTime = @belapsed CVXSolver(mK)  seconds = 2;
resAnalysis = @sprintf("The reference solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed ProximalGradientDescentAcc(vX, h∇f, hProjFun, η, numIter) setup = (vX = copy(vZZ)) seconds = 2;
resAnalysis = @sprintf("The PGM solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);