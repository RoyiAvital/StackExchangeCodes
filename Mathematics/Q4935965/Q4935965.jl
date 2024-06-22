# StackExchange Mathematics Q4935965
# https://math.stackexchange.com/questions/4935965
# Analytic Center Cutting Plane Method.
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  fd
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     22/06/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
using ECOS;
using FastLapackInterface; #<! Required for `JuliaOptimization.jl`
using LogExpFunctions;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = false;

oRng = StableRNG(1234);


## Functions


## Parameters

# Problem parameters
numRows = 10; #<! Matrix A
numCols = 8;  #<! Matrix A

# Solver Parameters
α               = 3.5e-0; #<! Step Size
numIterations   = Unsigned(50_000);


## Load / Generate Data
mA  = randn(oRng, numRows, numCols);
vX  = randn(oRng, numCols);
vB  = randn(oRng, numRows);
μ   = 1.0;
μ¹  = 1 / μ;

hObjFun( vX :: Vector{<: AbstractFloat} )   = μ¹ * LogExpFunctions.logsumexp(μ * (mA * vX - vB));
h∇Fun( vX :: Vector{<: AbstractFloat} )     = (1 / sum(exp.(μ * (mA * vX - vB)))) * (mA' * exp.(μ * (mA * vX - vB))); #<! Numerically instable

# Validate the Gradient Function
if (maximum(abs.(h∇Fun(vX) - CalcFunGrad(vX, hObjFun))) > 1e-6)
    println("The gradient function is not validated");
end


## Analysis

# Using DCP Solver (Convex.jl)
vXCvx = Variable(numCols);
sConvProb = minimize( μ¹ * Convex.logsumexp(μ * (mA * vXCvx - vB)) );
solve!(sConvProb, ECOS.Optimizer; silent = true);
vXRef = vec(vXCvx.value);
optVal = sConvProb.optval;

# Using Accelerated Gradient Descent

vXX = copy(vX);
vZ  = copy(vX);
vW  = copy(vX);
vObjFun = zeros(numIterations);
vObjFun[1] = hObjFun(vXX);

for ii ∈ 1:numIterations
    # FISTA (Nesterov) Accelerated

    ∇vZ = CalcFunGrad(vZ, hObjFun);

    vW .= vXX; #<! Previous iteration
    vXX .= vZ .- (α .* ∇vZ);

    fistaStepSize = (ii - 1) / (ii + 2);

    vZ .= vXX .+ (fistaStepSize .* (vXX .- vW));
    vObjFun[ii] = hObjFun(vXX);
end


## Display Results

figureIdx += 1;

# Using `height = nothing, width = nothing` means current size
oConf = PlotConfig(toImageButtonOptions = attr(format = "png", height = nothing, width = nothing).fields); #<! Won't work on VS Code

oTrace1 = scatter(x = 1:numIterations, y = vObjFun, mode = "lines", text = "Gradient Descent", name = "Gradient Descent",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = 1:numIterations, y = optVal * ones(numIterations), 
                  mode = "lines", text = "Optimal Value", name = "Optimum (Convex.jl)",
                  line = attr(width = 1.5, dash = "dot"));
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "Iteration", yaxis_title = "Value");
hP = plot([oTrace1, oTrace2], oLayout, config = oConf);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

