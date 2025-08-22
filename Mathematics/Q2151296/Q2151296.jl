# StackExchange Mathematics Q1160280
# https://math.stackexchange.com/questions/1160280
# Maximum Likelihood Estimator of Composition of Gaussian Random Variables with Background Noise.
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
# - 1.0.000     22/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Optim;
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

function LogLikelihood( mY :: Matrix{T}, vμ :: Vector{T}, mΣ :: Matrix{T}, vT :: Vector{T}, mT :: Matrix{T} ) where {T <: AbstractFloat}
    # https://stats.stackexchange.com/questions/351549 -> https://stats.stackexchange.com/a/391700

    paramM = size(mY, 2);

    mT .= zero(T);

    for ii in 1:paramM
        vT  .= view(mY, :, ii) .- vμ;
        mT .+= vT .* vT';
    end

    logLik = -paramM * logdet(mΣ) - tr(mT / mΣ);
    
    return logLik;    

end

function LogLikelihoodWrapper( vθ :: Vector{T}, mY :: Matrix{T}, vμ :: Vector{T}, mΣ :: Matrix{T}, vT :: Vector{T}, mT :: Matrix{T} ) where {T <: AbstractFloat}
    # vθ = [μ₁, μ₂, σ₁², σ₂², σᵦ²]

    vμ[1] = vθ[1];
    vμ[2] = vθ[2];
    mΣ[1] = vθ[3] + vθ[5];
    mΣ[2] = vθ[5];
    mΣ[3] = vθ[5];
    mΣ[4] = vθ[4] + vθ[5];

    logLik = LogLikelihood(mY, vμ, mΣ, vT, mT);

    return logLik;
    
end


## Parameters

# Data
numSamples = 250;
μ₁  = 0.23;
μ₂  = -0.11;
σ₁² = 0.55;
σ₂² = 0.36;
σᵦ² = 0.25;


## Load / Generate Data

vX₁ = μ₁ .+ sqrt(σ₁²) * randn(oRng, numSamples);
vX₂ = μ₂ .+ sqrt(σ₂²) * randn(oRng, numSamples);
vXᵦ = sqrt(σᵦ²) * randn(oRng, numSamples);

vY₁ = vX₁ + vXᵦ;
vY₂ = vX₂ + vXᵦ;

mY = [vY₁'; vY₂']; #<! Each column is a sample


## Analysis

vμ = vec(mean(mY; dims = 2));
mC = (1.0 / numSamples) * (mY .- vμ) * (mY .- vμ)';

mA = [1.0 0.0 1.0; 0.0 1.0 1.0; 0.0 0.0 1.0];
vB = [mC[1, 1], mC[2, 2], mC[1, 2]];
vθ = mA \ vB; #<! [σ₁², σ₂², σᵦ²]

vθθ = [vμ; vθ]; #<! [μ₁, μ₂, σ₁², σ₂², σᵦ²]

# Buffers
vMu = zeros(2);
mS  = zeros(2, 2);
vT  = zeros(2);
mT  = zeros(2, 2);

for ii in 1:numSamples
    vT  .= view(mY, :, ii) .- vμ;
    mT .+= vT .* vT';
end

hNegLogLik( vθ :: Vector{T} ) where {T <: AbstractFloat} = -LogLikelihoodWrapper(vθ, mY, vMu, mS, vT, mT);

# sOpt = optimize(hNegLogLik, ones(5)); #<! Works, yet not bounded
# sOpt = optimize(hNegLogLik, ones(5), LBFGS()); #<! Does not work
sOpt = optimize(hNegLogLik, [-10.0, -10.0, 1e-3, 1e-3, 1e-3], [10.0, 10.0, 10.0, 10.0, 10.0], ones(5), IPNewton()) #<! Works

# hObjFun = TwiceDifferentiable(hNegLogLik, ones(5); autodiff = :forward);
# sOpt = optimize(hObjFun, [-10.0, -10.0, 1e-3, 1e-3, 1e-3], [10.0, 10.0, 10.0, 10.0, 10.0], ones(5), IPNewton())

println("Direct method      : $(vθθ')");
println("Optimization method: $(sOpt.minimizer')");


## Display Results

# figureIdx += 1;

# vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, numα);

# for (ii, α) in enumerate(vα)
#     shiftVal = ii - 1.0;
#     nameStr = @sprintf("%0.2f", α);
#     vTr[ii] = scatter(; x = vY, y = shiftVal .+ mD[:, ii], mode = "lines", 
#                line_width = 1.5,
#                name = nameStr, text = nameStr);
# end

# sLayout = Layout(title = "Likelihood Function", width = 600, height = 600, 
#                  xaxis_title = "y", yaxis_title = "PDF",
#                  hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
#                  legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

# hP = Plot(vTr, sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end

