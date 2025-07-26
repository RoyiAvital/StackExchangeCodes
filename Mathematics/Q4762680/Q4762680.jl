# StackExchange Mathematics Q4762680
# https://math.stackexchange.com/questions/4762680
# Maximum Likelihood Parameter Estimation of Multinomial Distribution of 2 Binary Variables (Bivariate Binomial Distribution).
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  See [Multivariate Bernoulli Distribution](https://arxiv.org/abs/1206.1874).
#   3.  See [Bivariate Binomial Distribution](https://www.isroset.org/pub_paper/IJSRMSS/7-IJSRMSS-01310.pdf) (Includes correlation).
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     26/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using Distributions;
using ECOS;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
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

function LogLikelihood( vP :: Vector{T}, n0 :: N, n1 :: N, m0 :: N, m1 :: N, k00 :: N, k01 :: N, k10 :: N, k11 :: N ) where {T <: AbstractFloat, N <: Integer}

    vP  = copy(vP); #<! Keep original
    vP .= log.(vP);

    valL  = T(0);
    valL += vP[1] * k00 + vP[2] * k01 + vP[3] * k10 + vP[4] * k11; #<! k_ij
    valL += (vP[1] + vP[2]) * n0 + (vP[3] + vP[4]) * n1;
    valL += (vP[1] + vP[3]) * m0 + (vP[2] + vP[4]) * m1;

    return valL;
    
end

function CVXSolver( vQ :: Vector{T} ) where {T <: AbstractFloat}
    
    vP = Variable(4);
    
    sConvProb = minimize( -Convex.dot(vQ, Convex.log(vP)), [Convex.sum(vP) == 1, vP ≥ 0] );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vP.value);

end


## Parameters

# Data
numSamples1  = 15;
numSamples2  = 15;
numSamples12 = 75;

vP   = [2.0, 1.1, 4.3, 1.5]; #<! [p00, p01, p10, p11]
vP ./= sum(vP); #<! Normalize to sum 1

mE1 = [1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0];
mE2 = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0];


## Load / Generate Data

vT = rand(oRng, Multinomial(numSamples1, vP));
vN̂ = [vT[1] + vT[2], vT[3] + vT[4]];
vN = mE1' * (mE1 * vT);

vT = rand(oRng, Multinomial(numSamples2, vP));
vM̂ = [vT[1] + vT[3], vT[2] + vT[4]];
vM = mE2' * (mE2 * vT);

vK = rand(oRng, Multinomial(numSamples12, vP));


## Analysis

vP̂Ref = CVXSolver(vN + vM + vK);
println(vP);
println(vP̂Ref);

println(LogLikelihood(vP, vN̂[1], vN̂[2], vM̂[1], vM̂[2], vK[1], vK[2], vK[3], vK[4]));
println(dot(vN + vM + vK, log.(vP)));


## Display Results

# figureIdx += 1;

# sTr1 = contour(; x = vG, y = vG, z = mZ,
#                ncontours = 20,
#                autocontour = false,
#                contours_start = 0, contours_end = 5, contours_size = 0.25,
#                contours_coloring = "heatmap");
# sTr2 = scatter(; x = vXX, y = vYY, mode = "lines", 
#                line_width = 2.75,              
#                name = "Ellipse", text = "Ellipse");
# sTr3 = scatter(; x = vUU, y = vVV, mode = "lines", 
#                line_width = 2.75,              
#                name = "Unit Ball", text = "Unit Ball");
# sTr4 = scatter(; x = [vY[1]], y = [vY[2]], mode = "markers", 
#                marker_size = 7,
#                name = "Point to Project", text = "Point to Project");
# sTr5 = scatter(; x = [vXRef[1]], y = [vXRef[2]], mode = "markers", 
#                marker_size = 7,
#                name = "Projected Point (Ref)", text = "Projected Point (Ref))");
# sTr6 = scatter(; x = [vX[1]], y = [vX[2]], mode = "markers", 
#                marker_size = 7,
#                name = "Projected Point (Sol)", text = "Projected Point (Sol))");
# sLayout = Layout(title = "Objective Function", width = 600, height = 600, 
#                  xaxis_title = "x", yaxis_title = "y",
#                  xaxis_range = tuGrid, yaxis_range = tuGrid,
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
#                  legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = Plot([sTr1, sTr2, sTr3, sTr4, sTr5, sTr6], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end


