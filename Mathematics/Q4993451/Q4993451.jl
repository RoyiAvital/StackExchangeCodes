# StackExchange Mathematics Q4993451
# https://math.stackexchange.com/questions/4993451
# Solve Multiple Quadratic Equations.
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
# - 1.0.000     03/11/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
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

function f( vX :: Vector{T}, tA :: Array{T, 3}, vY :: Vector{T} ) where {T <: AbstractFloat}

    numEqs = size(tA, 3);

    f = zero(T);

    for ii ∈ 1:numEqs
        mA = view(tA, :, :, ii);
        f += (dot(vX, mA, vX) - vY[ii]) ^ 2;
    end

    return f;
    
end

function ∇f( vX :: Vector{T}, tA :: Array{T, 3}, vY :: Vector{T} ) where {T <: AbstractFloat}
    # Assumes `tA[:, :, ii]` is symmetric

    numElements = length(vX);
    numEqs = size(tA, 3);
    vG = zeros(T, numElements)

    for ii ∈ 1:numEqs
        mA = view(tA, :, :, ii);
        vG += T(4.0) * (dot(vX, mA, vX) - vY[ii]) * (mA * vX);
    end

    return vG;

end


## Parameters

numElements = 2; #<! Number of elements
numEqs      = 10; #<! Number of equations
ϵ           = 1e-5;

# Solver
numIter = 1_000;
η       = 1e-3;

# Visualization
numGridPts = 500;


## Load / Generate Data

vXRef = randn(oRng, numElements);
tA    = randn(oRng, numElements, numElements, numEqs); #<! Each slice on 3rd dimension is `mAi`
vY    = zeros(numEqs);

for ii ∈ 1:numEqs
    mA = view(tA, :, :, ii);
    # mA .+= mA';
    mA[:] = mA' * mA + I;
    mA .+= mA';
    vY[ii] = dot(vXRef, mA, vXRef);
end


## Analysis

hF( vX :: Vector{T} ) where {T <: AbstractFloat} = f(vX, tA, vY);
h∇f( vX :: Vector{T} ) where {T <: AbstractFloat} = ∇f(vX, tA, vY);

# Validate Gradient
vX = randn(oRng, numElements);
# @assert (maximum(abs.(h∇f(vX) - CalcFunGrad(vX, hF))) <= ϵ) "The gradient calculation is not verified";

vX = rand(numElements);
vX = GradientDescentBackTracking(vX, numIter, η, hF, h∇f);


mÂ = permutedims(reshape(tA, (numElements * numElements, numEqs)), (2, 1));
vX̂ = mÂ \ vY;
mR = reshape(vX̂, (numElements, numElements));
oSvdFac = svd(mR);
if (oSvdFac.S[1] / (oSvdFac.S[2] + 1e-6) > 4e5)
    println("Method worked");
    vXX = sqrt(oSvdFac.S[1]) * oSvdFac.U[:, 1];
    hF(vXX)
end



## Display Analysis

if (numElements == 2)
figureIdx += 1;
vG = LinRange(-2, 2, numGridPts);
mO = zeros(numGridPts, numGridPts);
vT = zeros(2);

for jj ∈ 1:numGridPts, ii ∈ 1:numGridPts
    # x, y notation (Matches `heatmap()`)
    vT[1] = vG[jj];
    vT[2] = vG[ii];
    mO[ii, jj] = hF(vT);
end

oTr = heatmap(; x = vG, y = vG, z = log1p.(mO));
oLayout = Layout(title = "Objective Function - Log Scale", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

end

