# StackExchange Mathematics Q1421999
# https://math.stackexchange.com/questions/1421999
# Solve Non Convex: minx∥Ax∥, s.t. ∥x∥=1 and Bx≥0.
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
# - 1.0.000     19/11/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using NNLS;
using SparseArrays;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function SolveProj( vY :: Vector{T}, sQPWork :: QPWorkspace{T} ) where {T <: AbstractFloat}

    sQPWork.status = :Unsolved;
    sQPWork.c .= .-vY;
    vX, _ = solve!(sQPWork);

    return vX;
    
end


## Parameters

# A matrix
numRowsA = 10;
numColsA = 5;

numRowsB = 7;
numColsB = numColsA;

# Solver
numIter = 10;
η       = 1e-9;

# Visualization
numGridPts = 500;


## Load / Generate Data

mA = randn(oRng, numRowsA, numColsA);
mB = randn(oRng, numRowsB, numColsB);


vC = zeros(numColsA);
vG = zeros(numRowsB);
mI = Matrix{Float64}(I(numColsA));

# Solving 0.5 * || x ||_2^2 - y^T x + 0.5 || y ||_2^2, s.t. B x >= 0
oQP         = QP(mI, vC, -mB, vG);
sWorkSpace  = QPWorkspace(oQP);


## Analysis

mAA = mA' * mA;

hF( vX :: Vector{T} ) where {T <: AbstractFloat}  = 0.5 * sum(abs2, (mA * vX));
h∇F( vX :: Vector{T} ) where {T <: AbstractFloat} = mAA * vX;
hSolveProj( vX :: Vector{T} ) where {T <: AbstractFloat} = SolveProj(vX, sWorkSpace);

# Initialization
vX = 100 * randn(oRng, numColsA);
vObjVal = zeros(numIter);

vObjVal[1] = hF(vX);

for ii ∈ 2:numIter
    vX .-= η .* h∇F(vX);
    for jj ∈ 1:10
        vX ./= norm(vX);
        vX  .= hSolveProj(vX);
    end

    vObjVal[ii] = hF(vX);
end


## Display Analysis

figureIdx += 1;

oTr = scatter(; x = 1:numIter, y = log1p.(vObjVal), mode = "lines", name = "Objective");
oLayout = Layout(title = "Objective Function - Log Scale", width = 600, height = 600, 
                 xaxis_title = "Iteration Index", yaxis_title = "Value [Log Scale]",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

