# StackExchange Mathematics Q4885452
# https://math.stackexchange.com/questions/4885452
# The Maximum Likelihood Estimator of N(θ + 2, θ²).
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
using Distributions;
using ForwardDiff;
using Optim;
using PlotlyJS;
using PolynomialRoots;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function LogLikelihood( θ :: S, vX :: AbstractVector{T} ) where{T <: AbstractFloat, S}

    numSamples = length(vX);
    L = -numSamples * log(θ * sqrt(2π)); #<! Log Likelihood
    sumItm = zero(T);
    for ii ∈ 1:numSamples
        sumItm += (vX[ii] - θ - 2) ^ 2;
    end
    L -= (sumItm / (2 * θ * θ));

    return -L;
    
end


## Parameters

# Problem parameters
numSamples  = 10_000;
θ           = 0.27;
numGridPts  = 100_000;

# Solver Parameters

## Load / Generate Data
vX = (θ + 2.0) .+ θ * randn(oRng, numSamples);

hLogLikelihood(θ) = LogLikelihood(θ, vX);

dSol = Dict();


## Analysis 

# Multi variate solvers require working with arrays
oConst = TwiceDifferentiableConstraints([0.001], [5.0]); #<! Box Constraints
oOpt   = optimize(vθ -> LogLikelihood(vθ[1], vX), oConst, [1.0], IPNewton(); autodiff = :forward);
θEst   = oOpt.minimizer[1];

dSol["MLE Optim Box Newton"] = (θEst, hLogLikelihood(θEst), abs(θEst - θ));

oOpt = optimize(θ -> hLogLikelihood(θ), 0.001, 5.0, Brent());
θEst = oOpt.minimizer;

dSol["MLE Optim Brent"] = (θEst, hLogLikelihood(θEst), abs(θEst - θ));

oMle = fit_mle(Normal, vX);
θEst = ((oMle.μ - 2) + oMle.σ) / 2; #<! Averaging the 2 estimations
dSol["MLE Distribution"] = (θEst, hLogLikelihood(θEst), abs(θEst - θ));

valA = -numSamples;
valB = -sum(vX .- 2.0);
valC = sum(abs2, vX .- 2.0);
vR   = roots([valC, valB, valA]);
vS   = 1e12 * ones(2);

for ii ∈ 1:2
    estθ = real(vR[ii]);
    if estθ > 0.0
        vS[ii] = LogLikelihood(estθ, vX);
    end
end
θEst = real(vR[argmin(vS)]);
dSol["MLE Root Finding"] = (θEst, hLogLikelihood(θEst), abs(θEst - θ));


vΘ = LinRange(0.26, 0.28, numGridPts);
vL = hLogLikelihood.(vΘ);


## Display Results

figureIdx += 1;

# Using `height = nothing, width = nothing` means current size
oConf = PlotConfig(toImageButtonOptions = attr(format = "png", height = nothing, width = nothing).fields); #<! Won't work on VS Code

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSol) + 1);

vTr[1] = scatter(x = vΘ, y = vL, mode = "lines", text = "Log Likelihood", name = "Log Likelihood",
                  line = attr(width = 3.0));

for (ii, keyVal) in enumerate(keys(dSol))
    vTr[ii + 1] = scatter(x = [dSol[keyVal][1]], y = [dSol[keyVal][2]], mode = "markers", 
                          text = keyVal, name = keyVal, marker = attr(size = 10.0));
end

oShp = [vline(θ, line_color = "red", line_dash = "dashdot", line_width = 3.0)];
oAnn = [attr(x = θ, y = hLogLikelihood(θ), text = "Ground Truth", 
             xanchor = "left", yanchor = "bottom", showarrow = false,
             font_color = "red")];

oLayout = Layout(title = "Log Likelihood and Estimators", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "θ", yaxis_title = "Log Likelihood",
                 shapes = oShp, annotations = oAnn);
hP = plot(vTr, oLayout, config = oConf);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

