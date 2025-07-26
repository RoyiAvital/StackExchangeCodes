# StackExchange Mathematics Q874823
# https://math.stackexchange.com/questions/874823
# Orthogonal Projection with Weighted Norm (Ellipsoid Like).
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
# - 1.0.000     21/07/2025  Royi Avital
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
using SCS;                 #<! Seems to support more cases for Continuous optimization than ECOS
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

function GenEllipseData( majRadius :: T, minRadius :: T, centerX :: T, centerY :: T, θ :: T; numPts :: N = 250 ) where {T <: AbstractFloat, N <: Integer}

    vT = collect(LinRange(T(0), T(7), numPts));
    vX = @. centerX + majRadius * cos(vT) * cos(θ) + minRadius * sin(vT) * sin(θ);
    vY = @. centerY + majRadius * cos(vT) * sin(θ) - minRadius * sin(vT) * cos(θ);

    return vX, vY;
    
end

function GenEllipseMatrix( majRadius :: T, minRadius :: T, centerX :: T, centerY :: T, θ :: T ) where {T <: AbstractFloat}
    
    # Rotation matrix
    mR = [cos(θ) -sin(θ); sin(θ) cos(θ)];

    # Diagonal matrix with inverse squares of the radii
    mD = Diagonal([T(1) / majRadius ^ 2, T(1) / minRadius ^ 2]);

    # Compute A = R * D * R'
    mA = mR * mD * mR';
    mA = T(0.5) * (mA + mA'); #<! Ensure symmetry

    # Center
    vC = [centerX, centerY];

    return mA, vC;

end

function CVXSolver( mA :: Matrix{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    dataDim = size(mA, 1);
    
    vX = Variable(dataDim); #<! [a, b, c, d, e, f]
    
    sConvProb = minimize( Convex.quadform(vX - vY, mA; assume_psd = true), Convex.norm2(vX) <= T(1) );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);

end


## Parameters

# Data
dataDim    = 2;
tuGrid     = (-5.0, 5.0);
numGridPts = 1_000;

# Ellipse parameters
majRadius = 3.0;
minRadius = 2.0;
centerX   = 1.8;
centerY   = -0.125;
θ         = π / 6.0;
# θ         = 0.0;

# Noise

## Load / Generate Data

vXX, vYY = GenEllipseData(majRadius, minRadius, centerX, centerY, θ); #<! Ellipse
vUU, vVV = GenEllipseData(1.0, 1.0, 0.0, 0.0, 0.0); #<! Unit L2 Ball

mA, vC = GenEllipseMatrix(majRadius, minRadius, centerX, centerY, θ);

hEllVal( vX :: Vector{T} ) where {T <: AbstractFloat} = dot(vX - vC, mA, vX - vC);

vY = vC;
vAy = mA * vY;
hG( λ :: T ) where {T <: AbstractFloat} = sum(abs2, (mA + λ * I) \ vAy);

dSolvers = Dict();

vG = LinRange(tuGrid[1], tuGrid[2], numGridPts);
mZ = zeros(numGridPts, numGridPts);

for (jj, xx) in enumerate(vG)
    for (ii, yy) in enumerate(vG)
        mZ[ii, jj] = hEllVal([xx, yy]);
    end
end

vλ = LinRange(0.0, 100, 1000);
vVal = hG.(vλ);

## Analysis

λ = FindZeroBinarySearch(λ -> (hG(λ) - 1), 0.0, 30.0);
vX = (mA + λ * I) \ vAy;


## Display Results

figureIdx += 1;

sTr1 = contour(; x = vG, y = vG, z = mZ,
               ncontours = 20,
               autocontour = false,
               contours_start = 0, contours_end = 5, contours_size = 0.25,
               contours_coloring = "heatmap");
sTr2 = scatter(; x = vXX, y = vYY, mode = "lines", 
               line_width = 2.75,              
               name = "Ellipse", text = "Ellipse");
sTr3 = scatter(; x = vUU, y = vVV, mode = "lines", 
               line_width = 2.75,              
               name = "Unit Ball", text = "Unit Ball");
sTr4 = scatter(; x = [vY[1]], y = [vY[2]], mode = "markers", 
               marker_size = 7,
               name = "Point to Project", text = "Point to Project");
sTr5 = scatter(; x = [vXRef[1]], y = [vXRef[2]], mode = "markers", 
               marker_size = 7,
               name = "Projected Point (Ref)", text = "Projected Point (Ref))");
sTr6 = scatter(; x = [vX[1]], y = [vX[2]], mode = "markers", 
               marker_size = 7,
               name = "Projected Point (Sol)", text = "Projected Point (Sol))");
sLayout = Layout(title = "Objective Function", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "y",
                 xaxis_range = tuGrid, yaxis_range = tuGrid,
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot([sTr1, sTr2, sTr3, sTr4, sTr5, sTr6], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

sTr1 = scatter(; x = vλ, y = vVal, mode = "lines", 
               line_width = 2.75,              
               name = "g(λ)", text = "g(λ)");
lShapes = [hline(1.0; line_width = 0.75, line_dash = "dash", line_color = "green"), vline(λ; line_width = 0.75, line_dash = "dash", line_color = "red")];
annText = @sprintf("λ = %0.2f", λ);
lAnn    = [attr(text = annText, x = λ, y = 1, xanchor = "left", axref = "x", ayref = "y", ax = 0.5, ay = 1.5, font_size = 14, font_color = "red", showarrow = true, arrowwidth = 2, arrowsize = 1, arrowhead = 1)];
sLayout = Layout(title = "λ Multiplier Optimization", width = 600, height = 600, 
                 xaxis_title = "λ", yaxis_title = "g(λ)",
                 xaxis_range = (-0.1, 3), yaxis_range = (-0.1, 3),
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99),
                 shapes = lShapes, annotations = lAnn);

hP = Plot([sTr1], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

# Run Time Analysis
# runTime = @belapsed CVXSolver(mD) seconds = 2;
# resAnalysis = @sprintf("The Convex.jl (SCS) solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

# runTime = @belapsed JuMPSolver(mD) seconds = 2;
# resAnalysis = @sprintf("The JuMP.jl (SCS) solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

# runTime = @belapsed SolveADMM(mD) seconds = 2;
# resAnalysis = @sprintf("The ADMM Method solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

# runTime = @belapsed SolvePGD(mD) seconds = 2;
# resAnalysis = @sprintf("The PGD Method solution run time: %0.5f [Sec]", runTime);
# println(resAnalysis);

