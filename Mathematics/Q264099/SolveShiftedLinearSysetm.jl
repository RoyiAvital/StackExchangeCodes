# Solving Shifted Linear System with SPD Matrix and Non Trivial Matrix.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     17/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function GenData( numRows :: N ) where {N <: Integer}

    # Problem Data
    mA = randn(numRows, numRows);
    mA = mA' * mA;
    mA = 0.5 * (mA' + mA);
    mC = randn(numRows, numRows);
    mC = (mC' * mC) + 0.05 * I;
    mC = 0.5 * (mC' + mC); #<! SPD Matrix
    
    vB = randn(numRows);2
    
    # Decomposition
    mU = cholesky(mC).U;
    vD = mU' \ vB;
    mE = mU' \ mA / mU; #<! Symmetric (Can be SPD / SPSD as `mA`)
    sH = hessenberg(mE);
    sE = eigen(mE);

    return mA, mC, vB, mU, vD, mE, sH, sE;

end

function SolveDirect( mA :: Matrix{T}, mC :: Matrix{T}, α :: T,  vB :: Vector{T}, vX :: Vector{T} ) where {T <: AbstractFloat}

    sC = cholesky(mA + α * mC);
    ldiv!(vX, sC, vB);

    return vX;

end

function SolveHessenberg( sH :: H, α :: T, vD :: Vector{T}, mU :: UT, vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat, H <: Hessenberg, UT <: UpperTriangular}

    ldiv!(vY, sH + α * I, vD);
    ldiv!(vX, mU, vY);

    return vX;

end

function SolveEigen( sE :: E, α :: T, vD :: Vector{T}, mU :: UT, vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat, E <: Eigen, UT <: UpperTriangular}

    mul!(vX, sE.vectors', vD);
    vX ./= (sE.values .+ α);
    mul!(vY, sE.vectors, vX);
    ldiv!(vX, mU, vY);

    return vX;

end


## Parameters

# Data
numRows = 10;
# vR = collect(250:250:1000);
vR = collect(250:250:3000);

# Model
α  = 0.275;


## Load / Generate Data

mA, mC, vB, mU, vD, mE, sH, sE = GenData(numRows);
vY = zeros(numRows); #<! Buffer
vX = zeros(numRows); #<! Buffer


## Analysis

# Solve
vXRef = copy(SolveDirect(mA, mC, α, vB, vX));

# Test for correctness
# Hessenberg
vX = SolveHessenberg(sH, α, vD, mU, vX, vY);
println(norm(vX - vXRef));

# Eigen
vX = SolveEigen(sE, α, vD, mU, vX, vY);
println(norm(vX - vXRef));

mRunTime = zeros(length(vR), 3);

# Test for Run Time
for (ii, nn) in enumerate(vR)
    mA, mC, vB, mU, vD, mE, sH, sE = GenData(nn);
    vY = zeros(nn); #<! Buffer
    vX = zeros(nn); #<! Buffer

    mRunTime[ii, 1] = @belapsed SolveDirect($mA, $mC, $α, $vB, $vX) seconds = 0.3;
    mRunTime[ii, 2] = @belapsed SolveHessenberg($sH, $α, $vD, $mU, $vX, $vY) seconds = 0.3;
    mRunTime[ii, 3] = @belapsed SolveEigen($sE, $α, $vD, $mU, $vX, $vY) seconds = 0.3;
end

mRunTime *= 1e3; #<! Sec -> Mili Sec

## Display Analysis

figureIdx += 1;

sTr1 = scatter(x = vR, y = mRunTime[:, 1], mode = "lines", text = "Direct Solver", name = "Direct Solver",
                line_width = 2);
sTr2 = scatter(x = vR, y = mRunTime[:, 2], mode = "lines", text = "Hessenberg Solver", name = "Hessenberg Solver",
                line_width = 2);
sTr3 = scatter(x = vR, y = mRunTime[:, 3], mode = "lines", text = "Eigen Solver", name = "Eigen Solver",
                line_width = 2);
sLayout = Layout(title = "Solver Run Time per Matrix Size", width = 600, height = 600, hovermode = "closest",
                xaxis_title = "Number of Rows", yaxis_title = "Run Time [Mili Sec]",
                legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));

hP = plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

