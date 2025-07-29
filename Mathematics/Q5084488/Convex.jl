# StackExchange Mathematics Q5084488
# https://math.stackexchange.com/questions/5084488
# Robust Method to Fit an Ellipse in R².
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
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCS;                 #<! Seems to support more cases for Continuous optimization than ECOS
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

function CVXSolver( mA :: Matrix{T}, upBound :: T ) where {T <: AbstractFloat}

    numSamples = size(mA, 1);
    numClasses = size(mA, 2);
    
    vX = Variable(numSamples, Positive());
    vY = Variable(numClasses, Positive());
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = Convex.maximize( Convex.entropy(vY), [mA' * vX == vY, vX >= 1, vX <= upBound] );
    Convex.solve!(sConvProb, SCS.Optimizer; silent = true);
    
    return vec(vX.value);

end


## Parameters

# Data
fileUrl = raw"https://raw.githubusercontent.com/FixelAlgorithmsTeam/FixelCourses/refs/heads/master/DataSets/ClassBalancing.csv";

# Model
upBound = 50.0;

## Load / Generate Data

mA = readdlm(download(fileUrl), ',', Float64);


## Analysis

vX = CVXSolver(mA, upBound);


## Display Results

# figureIdx += 1;

# sTr1 = scatter(; x = vX, y = vY, mode = "markers", 
#                marker_size = 7,
#                name = "Samples", text = "Samples");
# sTr2 = scatter(; x = vXX, y = vYY, mode = "lines", 
#                line_width = 2.75,              
#                name = "Estimated Ellipse", text = "Estimated Ellipse");
# sLayout = Layout(title = "Ellipse Fit", width = 600, height = 600, 
#                  xaxis_title = "x", yaxis_title = "y",
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
#                  legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = Plot([sTr1, sTr2], sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end

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

