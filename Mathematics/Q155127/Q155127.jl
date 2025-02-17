# StackExchange Mathematics Q155127
# https://math.stackexchange.com/questions/155127
# Calculate the Maximal Volume Ellipsoid (MVE) in a Polyhedron.
# References:
#   1.  Yin Zhang and Liyan Gao - On Numerical Solution of the Maximum Volume Ellipsoid Problem.
#   2.  Jianzhe Zhen, Dick den Hertog - Computing the Maximum Volume Inscribed Ellipsoid of a Polytopic Projection.
#   3.  Stephen Boyd, Lieven Vandenberghe - Convex Optimization, Geometric Problems (Center of Polytope / Polyhedron).
#   4.  Largest Ellipse Inscribed in a Polygon - Brute Force (https://discourse.mcneel.com/t/128264).
#   5.  [Xin Li - Numerical Methods for Engineering Design and Optimization](https://users.ece.cmu.edu/~xinli/classes/cmu_18660/Lec14.pdf).
#   6.  Linus Kallberg - Minimum Enclosing Balls and Ellipsoids in General Dimensions (https://www.es.mdu.se/pdf_publications/5687.pdf).
#   7.  Find a Large Inscribed Ellipsoid - Points Cloud, within Convex Hull, Contain No Point (https://github.com/hongkai-dai/large_inscribed_ellipsoid).
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     16/02/2025  Royi Avital
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
using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using Polyhedra;           #<! Working with Polytope / Polyhedron
using SparseArrays;
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

function SolveMVE( vX0 :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T}; numIter :: N = 100, ϵ :: T = T(1e-6) ) where {T <: AbstractFloat, N <: Integer}
    # Assumptions:
    # - `mA` - Full rank.
    # - The Polytope (Bounded Polyhedron) has interior.
    # - numCols < numRows << numCols ^ 2.

    numRows = size(mA, 1);
    numCols = size(mA, 2);
    normB   = norm(vB);

    minμ = T(1e-8);
    τ0   = T(0.75);

    vBAx = vB - mA * vX0;
    if any(vBAx .< zero(T))
        throw(DomainError(vX0, "Must be interior point of the Polyhedron"));
    end

    # Scaling of the Polyhedron
    mA = inv.(vBAx) .* mA; #<! Scaling the rows (Equivalent of `diagm(inv.(vB)) * mA`)
    vB = ones(T, numRows);

    vX = zeros(T, numCols);
    vY = ones(T, numRows);
    vBAx = copy(vB);
    vADx = copy(vB);
    vZ = copy(vB);

    αᵢ  = one(T);
    mEE = zeros(T, numCols, numCols);

    # @printf("\n  Residuals:         Primal     Dual    Duality  logdet(E)\n");
    # @printf("------------------------------------------------------------\n");

    for ii ∈ 1:numIter

        if ii > 1
            vBAx -= αᵢ * vADx;
            # vBAx = vBAx - astep * vADx;
        end
        
        mY  = diagm(vY);
        mEE = inv(mA' * mY * mA);
        mQ  = mA * mEE * mA';
        vH  = sqrt.(diag(mQ));
        
        if ii == 1
            valT = minimum(vBAx ./ vH);
            # valT = minimum(ii -> vBAx[ii]/ vH[ii], 1:numRows);
            vY ./= (valT * valT);
            vH  *= valT;
            vZ   = max.(T(0.1), vBAx - vH);
            mQ  *= (valT * valT);
            mY  /= (valT * valT);
        end

        vYZ = vY .* vZ;
        vYH = vY .* vH;

        dualGap = sum(vYZ) / numRows;

        rμ = min(T(0.5), dualGap) * dualGap;
        rμ = max(rμ, minμ);

        vR1 = -mA' * vYH;
        vR2 = vBAx - vH - vZ;
        vR3 = rμ .- vYZ;

        r1 = maximum(abs, vR1); #<! Dual gap
        r2 = maximum(abs, vR2); #<! Primal gap
        r3 = maximum(abs, vR3); #<! Duality

        resVal = max(r1, r2, r3);
        
        # objVal = T(0.5) * logdet(mEE);
        # @printf("  Iteration: %3i  ", ii);
        # @printf("%9.3f %9.3f %9.3f  %9.3f\n", r2, r1, r3, objVal);

        # The `(rμ <= minμ)` seems not well defined as `rμ = max(rμ, minμ)`.
        if (resVal < (ϵ * (one(T) + normB))) && (rμ <= minμ)
            # Converged
            vX += vX0;
            break;
        end

        mYQ   = mY * mQ;
        mYQQY = mYQ .* mYQ';
        vYH2  = T(2.0) * vYH;
        mYA   = mY * mA;
        mG    = mYQQY + diagm(max.(T(1e-12), vYH2 .* vZ));
        mT    = mG \ ((vH + vZ) .* mYA); #<! Equivalent of `mG \ (diagm(vH + vZ) * mY)`
        mATP  = ((vYH2 .* mT) - mYA)'; #<! Equivalent of `(diagm(vYH2) * mT) - mYA`

        vR3Dy = vR3 ./ vY;
        vR23  = vR2 - vR3Dy;
        vDx   = (mATP * mA) \ (vR1 + mATP * vR23);
        vADx  = mA * vDx;
        vDyDy = mG \ (vYH2 .* (vADx - vR23));
        vDy   = vY .* vDyDy;
        vDz   = vR3Dy - (vZ .* vDyDy);

        ax = -inv(min(minimum(-vADx ./ vBAx), -T(0.5)));
        # ax = -inv(min(minimum(ii -> -vADx[ii]/ vBAx[ii], 1:numRows), -T(0.5)));
        ay = -inv(min(minimum(vDyDy), -T(0.5)));
        az = -inv(min(minimum(vDz ./ vZ), -T(0.5)));
        # az = -inv(min(minimum(ii -> -vDz[ii]/ vZ[ii], 1:numRows), -T(0.5)));
        τ  = max(τ0, one(T) - resVal);
        
        αᵢ = τ * min(one(T), ax, ay, az);

        vX += αᵢ * vDx;
        vY += αᵢ * vDy;
        vZ += αᵢ * vDz;

    end

    return vX, mEE;

    
end

function FindInteriorPointPolyHedron( mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}
    # Solve:
    # max             t
    # subject to Ax + t1 <= b

    numRows = size(mA, 1);
    numCols = size(mA, 2);

    valT = Variable();
    vX   = Variable(numCols);

    sConvProb = maximize( valT, [mA * vX + valT <= vB] );
    solve!(sConvProb, ECOS.Optimizer; silent = true);

    return sConvProb.status, vec(vX.value);

end


function GenPolygon( mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    sPolyHedron = polyhedron(hrep(mA, vB)); #<! Building a polyhedron
    vrep(sPolyHedron); #<! Generating its vertices representation
    # Vertices are not ordered to generate the Convex Hull.
    vConvHull = Polyhedra.planar_hull(sPolyHedron.vrep) #<! Vector of 2D Points
    # mConvHull = reduce(hcat, vConvHull.points.points); #<! Matrix 2 * p
    mConvHull = stack(vConvHull.points.points; dims = 2); #<! Matrix 2 * p
    mConvHull = hcat(mConvHull, mConvHull[:, 1]); #<! Connect the last point to the first

    return mConvHull;

end

function GenEllipse( vX :: Vector{T}, mE :: Matrix{T}; numGridPts :: N = 2_000 ) where {T <: AbstractFloat, N <: Integer}

    vGrid = LinRange(0, 2π, numGridPts);
    mEll  = vX .+ mE * hcat(sin.(vGrid), cos.(vGrid))';

    return mEll;

end


## Parameters

dataFile = "data2d.mat"; #<! From https://www.cmor-faculty.rice.edu/~zhang/mve/index.html


## Load / Generate Data

dMat = matread(dataFile);
mA   = dMat["A"];
vB   = vec(dMat["b"]);
vX0  = vec(dMat["x0"]);


## Analysis

# Interior Point of the Polyhedron
# solverStatus, vX0 = FindInteriorPointPolyHedron(mA, vB); #<! `solverStatus` Is an `Enum`
# if Int(solverStatus) != 1
#     error("Interior point was not found!")
# end

# Infiltrator.toggle_async_check(false);
# Infiltrator.clear_disabled!();
vX, mEE = SolveMVE(vX0, mA, vB);
sCholE  = cholesky(mEE);
mE      = collect(sCholE.L);


## Display Analysis

mPolyTope = GenPolygon(mA, vB);
mEll      = GenEllipse(vX, mE);

figureIdx += 1;

oTr1 = scatter(; x = mPolyTope[1, :], y = mPolyTope[2, :], mode = "markers+lines", 
              line_width = 2,
              name = "Polygon", text = ["x = $(mPolyTope[1, ii]), y = $(mPolyTope[2, ii])" for ii ∈ 1:size(mPolyTope, 2)]);
oTr2 = scatter(; x = [vX0[1]], y = [vX0[2]], mode = "scatter", 
              marker_size = 12,
              name = "Interior Point");
oTr3 = scatter(; x = [vX[1]], y = [vX[2]], mode = "scatter", 
              marker_size = 12,
              name = "Ellipse Center");
oTr4 = scatter(; x = mEll[1, :], y = mEll[2, :], mode = "lines", 
              line_width = 2,
              name = "Ellipse", text = ["x = $(mEll[1, ii]), y = $(mEll[2, ii])" for ii ∈ 1:size(mEll, 2)]);
oLayout = Layout(title = "Maximum Volume Ellipse in a Polygon", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2, oTr3, oTr4], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

