# StackExchange Mathematics Q3853102
# https://math.stackexchange.com/questions/3853102
# Find the Maximal Area Circle in a Polygon Using Chebyshev Center.
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
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using Polyhedra;           #<! Working with Polytope / Polyhedron / Polygon
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

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

function FindMaximumAreaCircle( mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    numHalfSpaces = size(mA, 1);

    vC   = Variable(2); #<! Center
    valR = Variable(); #<! Radius

    vV        = [dot(vC, mA[ii, :]) + valR * norm(mA[ii, :]) <= vB[ii] for ii ∈ 1:numHalfSpaces]; #<! Constraints
    sConvProb = maximize( valR, vcat(vV...) ); #<! https://github.com/jump-dev/Convex.jl/issues/722
    
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vC.value), valR.value[1];

end

function IsPointInPolygon( vX :: Vector{T}, mA :: Matrix{T}, vB :: Vector{T} ) where {T <: AbstractFloat}
    
    pointIn = all(mA * vX .<= vB);

    return pointIn;
    
end

function DistancePointLine( vX :: Vector{T}, vP :: Vector{T}, valC :: T ) where {T <: AbstractFloat}

    return abs(dot(vX, vP) + valC) / norm(vP);
    
end


## Parameters

# Polygon (2D)
mA = [1.0 0.0; 0.0 1.0; -1.0 1.0; 1.0 -1.0; -1.0 -1.0; 1.0 1.0];
vB = [11.0, 10.0, 6.0, 7.0, -1.0, 18.0];

# Grid
tuGridX = (-3, 12);
tuGridY = (-4, 11);
numGridPts = 1_000;


## Load / Generate Data

numHalfSpaces = size(mA, 1);

mG = zeros(numGridPts, numGridPts);
vX = LinRange(tuGridX[1], tuGridX[2], numGridPts);
vY = LinRange(tuGridY[1], tuGridY[2], numGridPts);

vPt = zeros(2); #<! Point in 2D
vD  = zeros(numHalfSpaces);


## Analysis

# Circle
vC, valR = FindMaximumAreaCircle(mA, vB);

# Distance to Polygon
for (jj, xx) ∈ enumerate(vX)
    vPt[1] = xx;
    for (ii, yy) ∈ enumerate(vY)
        vPt[2] = yy;
        if !(IsPointInPolygon(vPt, mA, vB))
            continue;
        end
        for kk ∈ 1:numHalfSpaces
            vD[kk] = DistancePointLine(vPt, mA[kk, :], -vB[kk]);
        end
        mG[ii, jj] = minimum(vD);
    end
end


## Display Analysis

mPolyTope = GenPolygon(mA, vB);

figureIdx += 1;

sTr1 = scatter(; x = mPolyTope[1, :], y = mPolyTope[2, :], mode = "markers+lines", 
              line_width = 2,
              name = "Polygon", text = ["x = $(mPolyTope[1, ii]), y = $(mPolyTope[2, ii])" for ii ∈ 1:size(mPolyTope, 2)]);
sTr2 = scatter(; x = [vC[1]], y = [vC[2]], mode = "scatter", 
              marker_size = 12,
              name = "Circle Center");
sShape = circle(x0 = vC[1] - valR, y0 = vC[2] - valR, x1 = vC[1] + valR, y1 = vC[2] + valR);
sLayout = Layout(title = "Maximum Area Circle in a Polygon", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 xaxis_range = tuGridX, yaxis_range = tuGridY,
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01),
                 shapes = [sShape]);
hP = Plot([sTr1, sTr2], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

sTr1 = heatmap(x = vX, y = vY, z = mG, showscale = false, colorscale = "RdBu");
sTr2 = scatter(; x = mPolyTope[1, :], y = mPolyTope[2, :], mode = "markers+lines", 
              line_width = 2,
              name = "Polygon", text = ["x = $(mPolyTope[1, ii]), y = $(mPolyTope[2, ii])" for ii ∈ 1:size(mPolyTope, 2)]);
sLayout = Layout(title = "Distance Transform to Polygon", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 xaxis_range = tuGridX, yaxis_range = tuGridY,
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));
hP = Plot([sTr1, sTr2], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

