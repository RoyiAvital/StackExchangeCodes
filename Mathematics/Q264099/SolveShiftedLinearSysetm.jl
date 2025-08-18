# Solving Shifted Linear System with SPD Matrix and Non Trivial Matrix.
#
# Release Notes
# - 1.1.000     18/08/2025  Royi Avital
#   *   Added QZ Decomposition based solver.
#   *   Measuring the Pre Process run time.
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

    return mA, mC, vB;

end

function PreProcessDirect( vB :: Vector{T} ) where {T <: AbstractFloat}

    vX = zero(vB);

    return vX;

end

function PreProcessHessenberg( mA :: Matrix{T}, mC :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    mU = cholesky(mC).U;
    vD = mU' \ vB;
    mE = mU' \ mA / mU; #<! Symmetric (Can be SPD / SPSD as `mA`)
    sH = hessenberg(mE);

    return mU, vD, mE, sH;

end

function PreProcessEigen( mA :: Matrix{T}, mC :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    mU = cholesky(mC).U;
    vD = mU' \ vB;
    mE = mU' \ mA / mU; #<! Symmetric (Can be SPD / SPSD as `mA`)
    sE = eigen(mE);

    return mU, vD, mE, sE;

end

function PreProcessQZ( mA :: Matrix{T}, mC :: Matrix{T} ) where {T <: AbstractFloat}

    mH, mT, mQ, mZ = schur(mA, mC);
    mH = UpperHessenberg(mH);
    mT = UpperTriangular(mT);
    mW = UpperHessenberg(zero(mA));

    return mH, mT, mQ, mZ, mW;

end

function GenDataAll( numRows :: N ) where {N <: Integer}

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

function SolveQZ( mH :: H, mT :: UT, mQ :: Matrix{T}, mZ :: Matrix{T}, α :: T, vB :: Vector{T}, vX :: Vector{T}, vY :: Vector{T}, mW :: H ) where {T <: AbstractFloat, H <: UpperHessenberg, UT <: UpperTriangular}

    mul!(vX, mQ', vB);
    mW .= mH .+ α .* mT;
    ldiv!(vY, mW, vX);
    mul!(vX, mZ, vY);

    return vX;

end


## Parameters

# Data
numRows = 10;
# vR = collect(250:250:1000);
vR = collect(250:250:3500);

# Model
α  = 0.275;


## Load / Generate Data

mA, mC, vB, mU, vD, mE, sH, sE = GenDataAll(numRows);
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

# QZ
mH, mT, mQ, mZ, mW = PreProcessQZ(mA, mC);
vX = SolveQZ(mH, mT, mQ, mZ, α, vB, vX, vY, mW);
println(norm(vX - vXRef));

mRunTimePre   = zeros(length(vR), 4); #<! Pre Process
mRunTimeSolve = zeros(length(vR), 4); #<! Solving

# Test for Pre Process Run Time
for (ii, nn) in enumerate(vR)
    mA, mC, vB = GenData(nn);

    mRunTimePre[ii, 1] = @belapsed PreProcessDirect($vB) seconds = 0.3;
    mRunTimePre[ii, 2] = @belapsed PreProcessHessenberg($mA, $mC, $vB) seconds = 0.3;
    mRunTimePre[ii, 3] = @belapsed PreProcessEigen($mA, $mC, $vB) seconds = 0.3;
    mRunTimePre[ii, 4] = @belapsed PreProcessQZ($mA, $mC) seconds = 0.3;
end

# Test for Solve Run Time
for (ii, nn) in enumerate(vR)
    mA, mC, vB, mU, vD, mE, sH, sE = GenDataAll(nn);
    mH, mT, mQ, mZ, mW = PreProcessQZ(mA, mC);
    vY = zeros(nn); #<! Buffer
    vX = zeros(nn); #<! Buffer

    mRunTimeSolve[ii, 1] = @belapsed SolveDirect($mA, $mC, $α, $vB, $vX) seconds = 0.3;
    mRunTimeSolve[ii, 2] = @belapsed SolveHessenberg($sH, $α, $vD, $mU, $vX, $vY) seconds = 0.3;
    mRunTimeSolve[ii, 3] = @belapsed SolveEigen($sE, $α, $vD, $mU, $vX, $vY) seconds = 0.3;
    mRunTimeSolve[ii, 4] = @belapsed SolveQZ($mH, $mT, $mQ, $mZ, $α, $vB, $vX, $vY, $mW) seconds = 0.3;
end

mRunTimeSolve *= 1e3; #<! Sec -> Mili Sec

## Display Analysis

figureIdx += 1;

sTr1 = scatter(x = vR, y = mRunTimePre[:, 1], mode = "lines", text = "Direct Solver", name = "Direct Solver",
                line_width = 2);
sTr2 = scatter(x = vR, y = mRunTimePre[:, 2], mode = "lines", text = "Hessenberg Solver", name = "Hessenberg Solver",
                line_width = 2);
sTr3 = scatter(x = vR, y = mRunTimePre[:, 3], mode = "lines", text = "Eigen Solver", name = "Eigen Solver",
                line_width = 2);
sTr4 = scatter(x = vR, y = mRunTimePre[:, 4], mode = "lines", text = "QZ Solver", name = "QZ Solver",
                line_width = 2);
sLayout = Layout(title = "Pre Process Run Time per Matrix Size", width = 600, height = 600, hovermode = "closest",
                xaxis_title = "Number of Rows", yaxis_title = "Run Time [Sec]",
                legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));

hP = plot([sTr1, sTr2, sTr3, sTr4], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

sTr1 = scatter(x = vR, y = mRunTimeSolve[:, 1], mode = "lines", text = "Direct Solver", name = "Direct Solver",
                line_width = 2);
sTr2 = scatter(x = vR, y = mRunTimeSolve[:, 2], mode = "lines", text = "Hessenberg Solver", name = "Hessenberg Solver",
                line_width = 2);
sTr3 = scatter(x = vR, y = mRunTimeSolve[:, 3], mode = "lines", text = "Eigen Solver", name = "Eigen Solver",
                line_width = 2);
sTr4 = scatter(x = vR, y = mRunTimeSolve[:, 4], mode = "lines", text = "QZ Solver", name = "QZ Solver",
                line_width = 2);
sLayout = Layout(title = "Solver Run Time per Matrix Size", width = 600, height = 600, hovermode = "closest",
                xaxis_title = "Number of Rows", yaxis_title = "Run Time [Mili Sec]",
                legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));

hP = plot([sTr1, sTr2, sTr3, sTr4], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

