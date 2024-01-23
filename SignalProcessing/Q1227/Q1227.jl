# StackExchnage Signal Processing Q1227
# https://dsp.stackexchange.com/questions/1227
# Piece Wise Linear Fit without Known Knots.
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
# - 1.0.000     22/01/2024  Royi Avital RoyiAvital@yahoo.com
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
tuGridSupport   = (0, 10);
numGridPts      = 1000;
numSections     = 5;

noiseStd = 0.5;

# Model
polyDeg = 1;
λ       = 505.5;
vλ      = LinRange(0.1, 10, 5);
ρ       = 199.5; #<! Should be proportional to λ

# Solvers
numIterations = Unsigned(50_000);

## Generate / Load Data

# Data
numSamples = numGridPts;

vG = LinRange(tuGridSupport[1], tuGridSupport[2], numGridPts);
# mG = [vG[ii] ^ jj for ii in 1:numGridPts, jj in 0:1]; #<! Matrix model for affine function
mG = vG .^ (0:1)'; #<! Matrix model for affine function
# mP = randn(oRng, 2, numSections); #<! Parameters of linear function
# mP = mP[:, sortperm(mP[1, :])]; #<! First column a, second b : y_i = a x_i + b
# vY = vec(maximum(mG * mP; dims = 2) + (noiseStd * randn(oRng, numGridPts)));

vP = sort(5 * randn(oRng, numSections)); #!< Values
vP = [-7, 5, -4, 9, -2]; #!< Values
vI = reduce(vcat, [1, sample(oRng, 10:(numGridPts - 10), numSections - 1; replace = false, ordered = true), numGridPts]); #<! Break index
vI = [1, 188, 407, 570, 810, numGridPts]; #<! Break index
vY = zeros(numGridPts);

for ii in 1:numSections
    vY[vI[ii]:vI[ii + 1]] .= vP[ii];
end
vY = cumsum(vY);
vY = 5. * (vY ./ maximum(abs.(vY)));
vY .+= noiseStd * randn(oRng, numGridPts);


# Model
mD = spdiagm(numSamples, numSamples, -1 => -ones(numSamples - 1), 0 => ones(numSamples));
for kk in 1:polyDeg
    mD[:] = mD * mD;
end
mD = mD[(polyDeg + 2):end, :];

# Solver
mX = zeros(numGridPts, numIterations);


# See https://discourse.julialang.org/t/73206
hObjFun( vX :: Vector{T}, λ :: T ) where{T <: AbstractFloat} = 0.5 * sum(abs2, vX - vY) + λ * norm(mD * vX, 1);

dSolvers = Dict();

## Display Data

figureIdx += 1;

oTr = scatter(x = collect(vG), y = vY, 
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

vX = Variable(numGridPts);
sConvProb = minimize( 0.5 * sumsquares(vX - vY) + λ * norm(mD * vX, 1) );
solve!(sConvProb, ECOS.Optimizer; silent_solver = true);
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
oTr2 = scatter(x = vG, y = mX[:, end], 
                mode = "lines", text = "ADMM", name = "Convex.jl");
oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr1, oTr2], oLayout);
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