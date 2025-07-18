# StackExchange Mathematics Q3832410
# https://math.stackexchange.com/questions/3832410
# Efficient Solver for $\sum_{i = 1}^{n} {\left\| \boldsymbol{a}_{i} - \boldsymbol{x} \right\|}_{\infty}^{2}$ with Large $n$.
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
# - 1.0.000     14/07/2025  Royi Avital
#   *   First release.

## Packages
using Convex;
using ECOS;

## Functions

function SolveProblemConvex( mA :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim    = size(mA, 1);
    numSamples = size(mA, 2);

    vX = Variable(dataDim);

    vV        = [Convex.norm_inf(vX - mA[:, ii]) for ii ∈ 1:numSamples];
    # sConvProb = minimize( Convex.sum(vV) ); #<! Works
    sConvProb = minimize( Convex.sumsquares(vV) ); #<! Does not work
    
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vX.value;
    
end

## Parameters
dataDim    = 5;
numSamples = 10;

## Data
mA = randn(dataDim, numSamples);

## Analysis
vX = SolveProblemConvex(mA);
