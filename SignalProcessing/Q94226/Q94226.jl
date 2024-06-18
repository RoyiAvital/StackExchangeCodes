# StackExchnage Signal Processing Q94226
# https://dsp.stackexchange.com/questions/94226
# Detect and Localize Abrupt Changes on a Noisy Non Periodic Signal
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   3. 
# TODO:
# 	1.  C
# Release Notes
# - 1.0.000     18/06/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
using SparseArrays;
# External
using Convex;
using ECOS;
using FastLapackInterface;
using PlotlyJS;
using StableRNGs;
using StatsBase;


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


## Parameters

# Data
fileUrl = raw"https://gist.githubusercontent.com/cernejr/17ed740798c41b527c7ef086f3a2e3fd/raw/a72eb49aeb3c03e0ff55efd8a7127dbd58b58f3f/gistfile1.txt"

# Model
polyDeg = 0; #<! Degree of the fitted polynomial
λ       = 500.5;
ρ       = 199.5; #<! Should be proportional to λ

# Solvers
numIterations = Unsigned(50_000);

## Generate / Load Data

# Data
mData = readdlm(download(fileUrl), ',', Float64);

numSamples = size(mData, 1);
vG = mData[:, 1];
vY = mData[:, 2];


# Model
# Forward Finite Differences matrix
mD = spdiagm(numSamples, numSamples, 0 => -ones(numSamples), 1 => ones(numSamples - 1));
mDD = copy(mD);
for kk in 1:polyDeg
    mD[:] = mD * mDD;
end
mD = mD[1:(end - polyDeg - 1), :];

# Solver
mX = zeros(numSamples, numIterations);


# See https://discourse.julialang.org/t/73206
hObjFun( vX :: Vector{T}, λ :: T ) where{T <: AbstractFloat} = 0.5 * sum(abs2, vX - vY) + λ * norm(mD * vX, 1);

dSolvers = Dict();

## Display Data

figureIdx += 1;

oTr = scatter(x = vG, y = vY, 
                mode = "markers", text = "Data Samples", name = "Data Samples");
oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


## Analysis

# DCP Solver
methodName = "Convex.jl"

vX = Variable(numSamples);
sConvProb = minimize( 0.5 * sumsquares(vX - vY) + λ * norm(mD * vX, 1) );
solve!(sConvProb, ECOS.Optimizer; silent = true);
vXRef = vec(vX.value);
optVal = sConvProb.optval;
                       
dSolvers[methodName] = vXRef;

# ADMM
methodName = "ADMM";
mDD = I(numSamples) + ρ * (mD' * mD);
sDC = cholesky(mDD);

vZ = zeros(size(mD, 1));
vU = zeros(size(mD, 1));
hProxF(vS, λ) = sDC \ (vY + ρ * mD' * vS);
hProxG(vY, λ) = max.(abs.(vY) .- λ, 0) .* sign.(vY);

ADMM!(mX, vZ, vU, mD, hProxF, hProxG; ρ, λ);


dSolvers[methodName] = mX[:, end];


## Display Results

figureIdx += 1;

oTr1 = scatter(x = vG, y = vY, 
                mode = "markers", text = "Data Samples", name = "Data Samples");
oTr2 = scatter(x = vG, y = vXRef, 
                mode = "lines", text = "Convex.jl", name = "Convex.jl");
oTr3 = scatter(x = vG, y = mX[:, end], 
                mode = "lines", text = "ADMM", name = "ADMM");
oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr1, oTr2, oTr3], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers) + 1);

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii + 1] = scatter(x = vG, y = dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0));
end
vTr[1] = scatter(x = vG, y = vY, 
                mode = "markers", text = "Data Samples", name = "Data Samples");
oLayout = Layout(title = "Data Samples and Estimations", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


## Derivative

vD = mD * mX[:, end]

figureIdx += 1;

oTr = scatter(x = vG, y = abs.(vD), 
                mode = "markers", text = "Derivative Magnitude", name = "Derivative Magnitude");
oLayout = Layout(title = "Derivative Signal", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end