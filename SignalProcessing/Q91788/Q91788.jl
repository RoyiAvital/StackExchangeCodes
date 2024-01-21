# StackExchnage Signal Processing Q91788
# https://dsp.stackexchange.com/questions/91788
# Fit Data Samples with Large Number of Samples and Outliers with a Ronust Fit.
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
# - 1.0.000     21/12/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
using ECOS;
using FastLapackInterface;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
# include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions

function IRLS!( vX :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}, vW :: Vector{T}, mWA :: Matrix{T}, mC :: Matrix{T}, vT :: Vector{T}, sBKWorkSpace :: BunchKaufmanWs{T}; normP :: T = one(T), numItr :: N = 1000, ϵ :: T = T(1e-6) ) where {T <: AbstractFloat, N <: Unsigned}

    errThr = T(1e-6); #<! Should be adaptive per iteration
    effNorm = ((normP - T(2)) / T(2));
    
    for _ in 1:numItr
        mul!(vW, mA, vX);
        vW .-= vB; #<! Error
        # Basically solving (vW .* A) \ (vW .* vB) <-> (mA' * Diag(vW) * mA) \ (mA' * Diag(vW) * vB).
        # Assuming m << n (size(mA, 1) << size(mA, 2)) it is faster to solve the normal equations.
        # The cost is doubling the condition number.
        vW .= max.(abs.(vW), errThr) .^ effNorm;
        vW .= vW ./ sum(vW);
        mWA .= vW .* mA;
        mul!(mC, mWA', mWA); #<! (mWA' * mWA) 
        # mC .= 0.5 .* (mC .+ mC'); #<! Guarantees symmetry (Allocates, seems to protect from aliasing)
        # for jj in 2:size(mC, 2)
        #     for ii in 1:(jj - 1)
        #         mC[jj, ii] = mC[ii, jj];
        #     end
        # end
        # No need to symmetrize `mC` as the decomposition looks only on a single triangle
        vW .= vW .* vB; #<! (mW * vB);
        copy!(vT, vX); #<! Previous iteration
        mul!(vX, mWA', vW); #<! (mWA' * mW * vB);
        # ldiv!(cholesky!(mC), vX); #<! vX = (mWA' * mWA) \ (mWA' * mW * vB);
        # Using Bunch-Kaufman as it works for SPSD (Cholesky requires SPD).
        _, ipiv, _ = LAPACK.sytrf!(sBkWorkSpace, 'U', mC); #<! Applies the decomposition
        sBkFac = BunchKaufman(mC, ipiv, 'U', true, false, BLAS.BlasInt(0));
        ldiv!(sBkFac, vX); #<! vX = (mWA' * mWA) \ (mWA' * mW * vB);
        vT .= abs.(vX .- vT);
        if maximum(vT) <= ϵ
            break;
        end
    end

    return vX;
    
end

function IRLS(mA :: Matrix{T}, vB :: Vector{T}; normP :: T = one(T), numItr :: N = 1000 ) where {T <: AbstractFloat, N <: Unsigned}

    vX  = Vector{T}(undef, size(mA, 2));
    vT  = Vector{T}(undef, size(mA, 2));
    vW  = Vector{T}(undef, size(mA, 1));
    mWA = Matrix{T}(undef, size(mA));
    mC  = Matrix{T}(undef, size(mA, 2), size(mA, 2));
    sBkWorkSpace = BunchKaufmanWs(mC);

    vX = IRLS!(vX, mA, vB, vW, mWA, mC, vT, sBkWorkSpace; normP = normP, numItr = numItr);

    return vX;
    
end


## Parameters

# Data
vXFileUrl = "https://github.com/scipy/scipy/files/13993065/vX.csv";
vYFileUrl = "https://github.com/scipy/scipy/files/13993067/vY.csv";

# Model
modelNorm   = 1; #<! 1 <= modelNorm <= inf
polyDeg     = 1;

# Solvers
numIterations = Unsigned(50);

## Generate / Load Data
oRng = StableRNG(1234);

vX = vec(open(readdlm, download(vXFileUrl)));
vY = vec(open(readdlm, download(vYFileUrl)));

numSamples = size(vX, 1);


mX = [(vX[ii] ^ jj) for ii in 1:numSamples, jj in 0:1];


# See https://discourse.julialang.org/t/73206
hObjFun( vP :: Vector{<: AbstractFloat} ) = norm(mX * vP - vY);

dSolvers = Dict();


## Analysis

# DCP Solver
methodName = "Convex.jl"

vP0 = Variable(polyDeg + 1);
sConvProb = minimize(norm(mX * vP0 - vY, modelNorm));
solve!(sConvProb, ECOS.Optimizer; silent_solver = true);
vPRef = vec(vP0.value)
optVal = sConvProb.optval;

dSolvers[methodName] = vPRef;

# Iterative Reweighted Least Squares (IRLS)
methodName = "IRLS";

vP  = zeros(polyDeg + 1);
vT  = zeros(polyDeg + 1);
vW  = zeros(numSamples);
mWA = zeros(size(mX));
mC  = zeros(polyDeg + 1, polyDeg + 1);
sBkWorkSpace = BunchKaufmanWs(mC);

vP = IRLS!(vP, mX, vY, vW, mWA, mC, vT, sBkWorkSpace; normP = 1.0, numItr = numIterations);
# vP = IRLS(mX, vY; normP = 1.0, numItr = numIterations);

# @btime IRLS!(vP, mX, vY, vW, mWA, mC, vT, sBkWorkSpace; normP = 1.0, numItr = Unsigned(1000), ϵ = 0.0) setup=(vP = zeros(2))

dSolvers[methodName] = vP;

# Least Squares
methodName = "LS";

vP = mX \ vY;

dSolvers[methodName] = vP;


## Display Results

figureIdx += 1;

oTr = scatter(x = vX, y = vY, 
                mode = "markers", text = "Input Data", name = "Input Data");
oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers) + 1);

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
for (ii, methodName) in enumerate(keys(dSolvers))
    vTr[ii] = scatter(x = vX, y = mX * dSolvers[methodName], 
               mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0));
end
vTr[end] = scatter(x = vX, y = vY, 
                mode = "markers", text = "Input Data", name = "Input Data");
oLayout = Layout(title = "Data Samples and Estimations", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "y");

hP = plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end