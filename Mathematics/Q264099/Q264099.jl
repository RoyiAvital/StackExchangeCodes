# StackExchange Mathematics Q264099
# https://math.stackexchange.com/questions/264099
# Solving the Primal Kernel SVM Problem.
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
# - 1.0.000     12/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
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

function SolveCVX( mK :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # Olivier Chapelle - Training a Support Vector Machine in the Primal

    # `mK` the kernel matrix
    numSamples = size(mK, 1);

    vα     = Convex.Variable(numSamples);
    paramB = Convex.Variable(1);

    vH = [Convex.pos(T(1) - vY[ii] * (vα' * mK[:, ii] + paramB)) for ii in 1:numSamples];
    hingeLoss = Convex.sum(vcat(vH...));

    sConvProb = minimize( 0.5 * λ * Convex.quadform(vα, mK; assume_psd = true) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vα.value), paramB.value;
    
end


## Parameters

# Data
λ = 0.5;

# Solver


## Load / Generate Data

# From SK Learn Example (https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)
mX = [
     0.4 -0.7;
    -1.4 -0.9;
     1.5 -1.0;
    -1.3 -1.2;
    -1.1 -0.2;
    -1.2 -0.4;
    -0.5  1.2;
    -1.5  2.1;
     1.0  1.0;
     1.3  0.8;
     1.2  0.5;
     0.2 -2.0;
     0.5 -2.4;
     0.2 -2.3;
     0.0 -2.7;
     1.3  2.1;
];

vY = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];


## Analysis

mK = mX * mX';
vα, paramB = SolveCVX(mK, vY, λ);


## Display Analysis

figureIdx += 1;

# sTr1 = scatter(x = vT, y = vY, mode = "markers", text = "Data Samples", name = "Data Samples",
#                 marker = attr(size = 6));
# sTr2 = scatter(x = vT, y = CalcModel(vP0, vT), 
#                 mode = "lines", text = "Estimated Model (Initial), RMSE = ($initRmse)", name = "Estimated Model (Initial), RMSE = ($initRmse)",
#                 line = attr(width = 2.5));
# sTr3 = scatter(x = vT, y = CalcModel(vP, vT), 
#                 mode = "lines", text = "Estimated Model (Tuned), RMSE = ($tunedRmse)", name = "Estimated Model (Tuned), RMSE = ($tunedRmse)",
#                 line = attr(width = 2.5));
# sLayout = Layout(title = "The Data and Estimated Model", width = 600, height = 600, hovermode = "closest",
#                 xaxis_title = "t", yaxis_title = "y",
#                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = plot([sTr1, sTr2, sTr3], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end

