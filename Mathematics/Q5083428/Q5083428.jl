# StackExchange Mathematics Q5083428
# https://math.stackexchange.com/questions/5083428
# Force a Flat Column Sum of a Matrix.
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
# - 1.0.000     17/07/2025  Royi Avital
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
using GLPK; #<! MILP
using HiGHS; #<! MILP
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCIP; #<! MILP
using SCS;
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

function SolveContSmooth( mA :: Matrix{T} ) where {T <: AbstractFloat}

    numRows = size(mA, 1);
    numCols = size(mA, 2);

    vOnes = ones(T, numCols);
    
    valC = Variable();
    vX   = Variable(numRows);
    vE   = Variable(numCols);

    # lConvCons = [vX >= 1.0, mA' * vX + vE == valC * vOnes]; #<! Constraints
    lConvCons = [vX >= 1.0, mA' * vX + vE == valC]; #<! Constraints
    sConvProb = minimize( Convex.sum(abs(vE)) + valC, lConvCons ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    # Convex.solve!(sConvProb, SCS.Optimizer; silent = true);

    return vec(vX.value), valC.value[1];
    
end

function SolveDiscrete( mA :: Matrix{T}; maxVal :: T = T(100) ) where {T <: AbstractFloat}

    numRows = size(mA, 1);
    numCols = size(mA, 2);
    
    t  = Variable();
    μ  = Variable();
    vX = Variable(numRows, IntVar);

    lConvCons = [mA' * vX - μ <= t, μ - mA' * vX <= t, vX >= 1, vX <= maxVal, sum(mA' * vX) == numCols * μ]; #<! Constraints (Seems to be tighter)
    # lConvCons = [mA' * vX - μ <= t, μ - mA' * vX <= t, vX >= 1, vX <= maxVal]; #<! Constraints
    sConvProb = minimize( t, lConvCons ); #<! Problem
    # Convex.solve!(sConvProb, SCIP.Optimizer; silent = true);
    Convex.solve!(sConvProb, HiGHS.Optimizer; silent = true);

    return vec(vX.value), μ.value[1];
    
end


## Parameters

# Data
numRows = 40;
numCols = 8;
minVal  = 0;
maxVal  = 9;

# Solver


## Load / Generate Data

mA = Float64.(rand(oRng, minVal:maxVal, numRows, numCols));


## Analysis

println("Continuous Solution")
vX, _ = SolveContSmooth(mA);
vX   .= round.(vX);
μ     = mean(mA' * vX);
maxAbsDev = maximum(abs.(mA' * vX .- μ));
println(@sprintf("Maximum Absolute Deviation: %0.2f", maxAbsDev));
println(@sprintf("Variance: %0.2f", var(mA' * vX)));

println("Discrete Solution")
vX, _ = SolveDiscrete(mA);
vX   .= round.(vX);
μ     = mean(mA' * vX);
maxAbsDev = maximum(abs.(mA' * vX .- μ));
println(@sprintf("Maximum Absolute Deviation: %0.2f", maxAbsDev));
println(@sprintf("Variance: %0.2f", var(mA' * vX)));


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

