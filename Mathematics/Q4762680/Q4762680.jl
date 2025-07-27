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
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaProxOperators.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function LogLikelihood( vP :: Vector{T}, n0 :: N, n1 :: N, m0 :: N, m1 :: N, k00 :: N, k01 :: N, k10 :: N, k11 :: N ) where {T <: AbstractFloat, N <: Integer}

    vPl = log.(vP);

    valL  = T(0);
    valL += vPl[1] * k00 + vPl[2] * k01 + vPl[3] * k10 + vPl[4] * k11; #<! k_ij
    valL += log(vP[1] + vP[2]) * n0 + log(vP[3] + vP[4]) * n1;
    valL += log(vP[1] + vP[3]) * m0 + log(vP[2] + vP[4]) * m1;

    return valL;
    
end

function CVXSolver( vK :: Vector{T}, vN :: Vector{T}, vM :: Vector{T}, mE1 :: Matrix{T}, mE2 :: Matrix{T} ) where {T <: AbstractFloat}
    
    vP = Variable(4);
    
    sConvProb = maximize( Convex.dot(vK, Convex.log(vP)) + Convex.dot(vN, Convex.log(mE1 * vP)) + Convex.dot(vM, Convex.log(mE2 * vP)), [Convex.sum(vP) == 1, vP ≥ 0] );
    # Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    Convex.solve!(sConvProb, SCS.Optimizer; silent = true);
    
    return vec(vP.value);

end

function EM!( vP :: Vector{T}, n0 :: N, n1 :: N, m0 :: N, m1 :: N, k00 :: N, k01 :: N, k10 :: N, k11 :: N; numIter :: N = 1000 ) where {T <: AbstractFloat, N <: Integer}
    # Interpret the X1 only and X2 only observations as incomplete: each such observation has an unobserved partner variable.
    # EM alternates between imputing expected joint counts (E step) and maximizing as if we had complete data (M step).
    # p_i* = p_i0 + p_i1, p_*j = p_0j + p_1j

    vQ = zero(vP); #<! [q_00, q_01, q_10, q_11], virtual counts
    Nₜ = n0 + n1 + m0 + m1 + k00 + k01 + k10 + k11; #<! Total number of trials

    # Split n0​ between (0, 0) and (0, 1): n0 * p_00 / p_0*, n0 * p_01 / p_0*.
    # Split n1​ between (1, 0) and (1, 1): n1 * p_10 / p_1*, n1 * p_11 / p_1*.
    # Split m0​ between (0, 0) and (1, 0): m0 * p_00 / p_*0, m0 * p_10 / p_*0.
    # Split m1​ between (0, 1) and (1, 1): m1 * p_01 / p_*1, m1 * p_11 / p_*1.

    for _ ∈ 1:numIter
        # E Step
        vQ[1] = k00 + n0 * vP[1] / (vP[1] + vP[2]) + m0 * vP[1] / (vP[1] + vP[3]);
        vQ[2] = k01 + n0 * vP[2] / (vP[1] + vP[2]) + m1 * vP[2] / (vP[2] + vP[4]);
        vQ[3] = k10 + n1 * vP[3] / (vP[3] + vP[4]) + m0 * vP[3] / (vP[1] + vP[3]);
        vQ[4] = k11 + n1 * vP[4] / (vP[3] + vP[4]) + m1 * vP[4] / (vP[2] + vP[4]);

        # M Step
        vP .= vQ ./ Nₜ;
    end

    return vP;
    
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
vT = rand(Multinomial(numSamples1, vP));
vN = [vT[1] + vT[2], vT[3] + vT[4]];

vT = rand(oRng, Multinomial(numSamples2, vP));
vT = rand(Multinomial(numSamples2, vP));
vM = [vT[1] + vT[3], vT[2] + vT[4]];

vK = rand(oRng, Multinomial(numSamples12, vP));
vK = rand(Multinomial(numSamples12, vP));

hObjFun( vP :: Vector{T} ) where {T <: AbstractFloat} = LogLikelihood(vP, vN[1], vN[2], vM[1], vM[2], vK[1], vK[2], vK[3], vK[4]);
hGradFun( vP :: Vector{T} ) where {T <: AbstractFloat} = -((vK ./ vP) + (mE1' * (vN ./ (mE1 * vP))) + (mE2' * (vM ./ (mE2 * vP))));
# hProjFun( vP :: Vector{T} ) where {T <: AbstractFloat} = ProjSimplexBall(vP);
hProjFun( vP :: Vector{T} ) where {T <: AbstractFloat} = ProjectSimplexBall(vP);


## Analysis

vP̂Ref = CVXSolver(Float64.(vK), Float64.(vN), Float64.(vM), mE1, mE2);
vP̂EM  = (vK .+ 1) ./ (numSamples1 + numSamples2 + numSamples12); #<! Adding 1 to ensure initial probability of all cases above 0
vP̂EM  = EM!(vP̂EM, vN[1], vN[2], vM[1], vM[2], vK[1], vK[2], vK[3], vK[4]);
vP̂GD  = (vK .+ 1) ./ (numSamples1 + numSamples2 + numSamples12); #<! Adding 1 to ensure initial probability of all cases above 0
vP̂GD  = GradientDescent(vP̂GD, 100_000, 1e-4, hGradFun; ProjFun = hProjFun);

println(vP);
println(vP̂Ref);
println(vP̂EM);
println(vP̂GD);

println(hObjFun(vP));
println(hObjFun(vP̂Ref));
println(hObjFun(vP̂EM));
println(hObjFun(vP̂GD));

# println(LogLikelihood(vP, vN̂[1], vN̂[2], vM̂[1], vM̂[2], vK[1], vK[2], vK[3], vK[4]));
# println(dot(vN + vM + vK, log.(vP)));


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


