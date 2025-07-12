# StackExchange Mathematics Q4989670
# https://math.stackexchange.com/questions/4989670
# Curve Fit with 2 Exponential Terms.
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
# - 1.0.000     12/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using LsqFit;
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

function LevenbergMarquardt(
    hF :: Function,       #<! Model function: hF(vP, vX) -> Vector
    hJ :: Function,       #<! Jacobian function: hJ(vP, vX) -> Matrix
    vX :: Vector{T},      #<! Input (Independent variable) data points
    vY :: Vector{T},      #<! Measured output values
    vP :: Vector{T};      #<! Initial guess for parameters
    maxIter :: Int = 100, #<! Number of iterations
    ϵ :: T = 1e-6,        #<! Stopping threshold (Successive loss)
    λ :: T = 1e-3,        #<! Regularization factor
    λFctr :: T = 10.0,    #<! Regularization scaling factor (`λFctr` >= 1.0)
    ) where {T <: AbstractFloat}
    # Ensure `hJ(vP, vX)` returns a matrix of size `(length(vY), length(vP))`.

    lossPrev = 1e20;
    isConv   = false;

    for ii in 1:maxIter
        vR = hF(vP, vX) - vY;           #<! Residual vector
        mJ = hJ(vP, vX);                #<! Jacobian matrix
        lossCurr = sum(abs2, vR); #<! Loss function

        # Solve (JᵗJ + λI) δ = -Jᵗr
        mH = mJ' * mJ + λ * I; #<! Approximate Hessian
        vG = mJ' * vR;         #<! Gradient
        vδ = -mH \ vG;         #<! Parameter update

        # Candidate step
        vPCan = vP + vδ;
        vR = hF(vPCan, vX) - vY;
        lossCan = sum(abs2, vR);

        if lossCan < lossCurr
            # Accept the step (Improves residual)
            vP = vPCan;
            λ /= λFctr; #<! Decrease λ
            if abs(lossCurr - lossCan) < ϵ
                isConv = true;
                break;
            end
        else
            λ *= λFctr; #<! Increase λ (more conservative)
        end

        lossPrev = lossCurr;
    end

    return vP, isConv;
end


function CalcModel( vP :: Vector{T}, vT :: Vector{T} ) where {T <: AbstractFloat}
    # Model: {y}_{n} = a + b * exp(p * t) + c * exp(q * t)
    # The `vP` vector: `[a, b, c, p, q]`
    vY = vP[1] .+ vP[2] .* exp.(vP[4] .* vT) .+ vP[3] .* exp.(vP[5] .* vT);
    
    return vY;
    
end

function CalcModelJac( vP :: Vector{T}, vT :: Vector{T} ) where {T <: AbstractFloat}
    # Model: {y}_{n} = a + b * exp(p * t) + c * exp(q * t)
    # The `vP` vector: `[a, b, c, p, q]`
    # Ensure `hJ(vP, vX)` returns a matrix of size `(length(vT), length(vP))`.
    
    numParams = length(vP);
    numSamples = length(vT);

    mJ = zeros(T, numSamples, numParams);
    mJ[:, 1] .= one(T); #<! With respect to `a`
    mJ[:, 2] .= exp.(vP[4] .* vT); #<! With respect to `b`
    mJ[:, 3] .= exp.(vP[5] .* vT); #<! With respect to `c`
    mJ[:, 4] .= vP[2] .* vT .* exp.(vP[4] .* vT); #<! With respect to `p`
    mJ[:, 5] .= vP[3] .* vT .* exp.(vP[5] .* vT); #<! With respect to `q`
    
    return mJ;
    
end


## Parameters

fileName = "Data.csv"; #<! Matches `T`, `YN` in question code

# Solver
numGridPts = 20_001;


## Load / Generate Data

mData = readdlm(fileName, ',', Float64; skipstart = 1);
vT    = mData[:, 1];
vY    = mData[:, 2];

numSamples = size(mData, 1);


## Analysis

# Solving for Model: {y}_{n} = a + b * exp(p * t) + c * exp(q * t)
# Based on: https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales (Jean Jacquelin - Regression and Equations of Integrals)
# See SciKit Guess (https://scikit-guess.readthedocs.io) and Non Linear Least Squares Minimization and Curve Fitting for Python (`lmfit`)

vS  = zeros(numSamples);
vSS = zeros(numSamples);

for ii ∈ 2:numSamples
    vS[ii]  = vS[ii - 1] + 0.5 * (vY[ii] + vY[ii - 1]) * (vT[ii] - vT[ii - 1]);
    vSS[ii] = vSS[ii - 1] + 0.5 * (vS[ii] + vS[ii - 1]) * (vT[ii] - vT[ii - 1]);
end

# Model Matrix
mM = [dot(vSS, vSS)     dot(vSS, vS)     dot(vSS, vT .^ 2) dot(vSS, vT) sum(vSS);
      dot(vSS, vS)      dot(vS, vS)      dot(vS, vT .^ 2)  dot(vS, vT)  sum(vS);
      dot(vSS, vT .^ 2) dot(vS, vT .^ 2) sum(vT .^ 4)      sum(vT .^ 3) dot(vT, vT);
      dot(vSS, vT)      dot(vS, vT)      sum(vT .^ 3)      dot(vT, vT)  sum(vT);
      sum(vSS)          sum(vS)          dot(vT, vT)       sum(vT)      numSamples];

vZ = [dot(vSS, vY), dot(vS, vY), dot(vT .^ 2, vY), dot(vT, vY), sum(vY)];
vP = mM \ vZ; #<! Intermediate parameters (`A`, `B`, `C`, `D`, `E`)

# Calculating `p` and `q` in the model
valP = 0.5 * (vP[2] + sqrt(vP[2] * vP[2] + 4.0 * vP[1]));
valQ = 0.5 * (vP[2] - sqrt(vP[2] * vP[2] + 4.0 * vP[1]));

mM = [numSamples            sum(exp.(valP .* vT))         sum(exp.(valQ .* vT))
      sum(exp.(valP .* vT)) sum(exp.(2valP .* vT))        sum(exp.((valQ + valP).* vT))
      sum(exp.(valQ .* vT)) sum(exp.((valQ + valP).* vT)) sum(exp.(2valQ .* vT))]
vZ = [sum(vY), dot(exp.(valP .* vT), vY), dot(exp.(valQ .* vT), vY)];
vP = mM \ vZ; #<! [a, b, c]

# Model Parameters
vP0 = [vP[1], vP[2], vP[3], valP, valQ]; #<! [a, b, c, p, q]

# Reference (For validation of my code for `LevenbergMarquardt()`)
# sFit = curve_fit((vX, vP) -> CalcModel(vP, vX), (vX, vP) -> CalcModelJac(vP, vX), vT, vY, vP0);
# vP = sFit.param;

vP, isConv = LevenbergMarquardt(CalcModel, CalcModelJac, vT, vY, vP0);

initRmse  = @sprintf("%0.2f", sqrt(mean(abs2, CalcModel(vP0, vT) - vY)));
tunedRmse = @sprintf("%0.2f", sqrt(mean(abs2, CalcModel(vP, vT) - vY)));


## Display Analysis

figureIdx += 1;


sTr1 = scatter(x = vT, y = vY, mode = "markers", text = "Data Samples", name = "Data Samples",
                marker = attr(size = 6));
sTr2 = scatter(x = vT, y = CalcModel(vP0, vT), 
                mode = "lines", text = "Estimated Model (Initial), RMSE = ($initRmse)", name = "Estimated Model (Initial), RMSE = ($initRmse)",
                line = attr(width = 2.5));
sTr3 = scatter(x = vT, y = CalcModel(vP, vT), 
                mode = "lines", text = "Estimated Model (Tuned), RMSE = ($tunedRmse)", name = "Estimated Model (Tuned), RMSE = ($tunedRmse)",
                line = attr(width = 2.5));
sLayout = Layout(title = "The Data and Estimated Model", width = 600, height = 600, hovermode = "closest",
                xaxis_title = "t", yaxis_title = "y",
                legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

