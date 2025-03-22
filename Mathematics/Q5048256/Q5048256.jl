# StackExchange Mathematics Q5048256
# https://math.stackexchange.com/questions/5048256
# Maximum Likelihood of Sum of 2 Uniform Variables.
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
# - 1.0.000     22/03/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Distributions;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using QuadGK;
using SparseArrays;
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

function PdfSumUniform( valT :: T, X :: Uniform{T}, Y :: Uniform{T} ) where {T <: AbstractFloat}
    # Using the Convolution definition.

    valMin = minimum(X) + minimum(Y);
    valMax = maximum(X) + maximum(Y);

    if (valT > valMax) || (valT < valMin)
        return zero(T);
    end

    valMin = max(minimum(X), valT - maximum(Y));
    valMax = min(maximum(X), valT - minimum(Y));

    valInt, valErr = quadgk(x -> pdf(X, x) * pdf(Y, valT - x), valMin - T(1e-5), valMax + T(1e-5));
    
    return valInt;
    
end

function PdfSumUniform( valT :: T, X :: Uniform{T}, Y :: Uniform{T} ) where {T <: AbstractFloat}
    # Analytic Solution
    #                                                  
    #              │                  │               
    #              │                  │               
    #             x│xxxxxxxxxxxxxxxxxx│x              
    #            xx│                  │xx            
    #          xxx │                  │ xxx           
    #         xx   │                  │   xx         
    #       xxx    │                  │    xxx       
    #     xxx      │                  │      xxx      
    #    xx        │                  │        xx    
    # ──x──────────┼──────────────────┼──────────x──   
    #  a+c        c+b                a+d        b+d  
    #                                                  
    # Assuming X ~ U[a, b], Y ~ U[c, d] with (d - c) >= (b - a).
    
    # Ensuring the assumption for XX and YY.
    suppX = maximum(X) - minimum(X);
    suppY = maximum(Y) - minimum(Y);

    if suppX > suppY
        XX = Y;
        YY = X;
        suppX, suppY = suppY, suppX;
    else
        XX = X;
        YY = Y;
    end

    valMin = minimum(X) + minimum(Y); #<! a + c
    valMax = maximum(X) + maximum(Y); #<! b + d

    if (valT > valMax) || (valT < valMin)
        return zero(T);
    end

    valA = minimum(X);
    valB = maximum(X);
    valC = minimum(Y);
    valD = maximum(Y);

    valCenter = inv(suppY) #<! Should be: `inv(suppX) * inv(suppY) * suppX;`
    valSlope  = valCenter / suppX;

    if valT <= (valC + valB)
        valOut = valSlope * (valT - valMin);
    elseif valT <= (valA + valD)
        valOut = valCenter;
    else #<! `(valT <= (valB + valD))`
        valOut = valSlope * (valMax - valT);
    end
    
    return valOut;
    
end

function CalLogLikelihood( vZ :: Vector{T}, paramR :: T, valA :: T, valB :: T ) where {T <: AbstractFloat}

    numSamples = length(vZ);

    logLik = zero(T);

    for ii ∈ 1:numSamples
        logLik += log(T(2) * paramR - abs(vZ[ii] - valA - valB));
    end

    logLik -= T(2) * numSamples * log(T(2) * paramR);
    
    return logLik;
    
end


## Parameters

numSamples = 500;

paramA = 3.0;
paramB = 5.0;
paramR = 1.0;

# Solver
numGridPts = 20_001;


## Load / Generate Data

X = Uniform(paramA - paramR, paramA + paramR);
Y = Uniform(paramB - paramR, paramB + paramR);
hPdfZ(valT :: T) where {T <: AbstractFloat} = PdfSumUniform(valT, X, Y);
minG = min(paramA - paramR - 1.0 , paramA + paramB - 2paramR - 1.0);
maxG = max(paramB + paramR + 1.0, paramA + paramB + 2paramR + 1.0);
vG = LinRange(minG, maxG, numGridPts);
pdfX = pdf(X, vG);
pdfY = pdf(Y, vG);
pdfZ = hPdfZ.(vG);

# Samples
vX = rand(oRng, X, numSamples);
vY = rand(oRng, Y, numSamples);
vZ = vX + vY;


## Analysis

# The minimum value of R must match the data
minR = ceil(maximum(abs.(vZ .- paramA .- paramB)) / 2.0; digits = 3);
vRGrid = LinRange(minR, minR + 0.5, 1_000);
vR = [CalLogLikelihood(vZ, valR, paramA, paramB) for valR ∈ vRGrid];


## Display Analysis

figureIdx += 1;

mA = [pdfX;; pdfY;; pdfZ];

hP = PlotLine(collect(vG), mA; plotTitle = "The PDF's", vSigNames = ["X", "Y", "Z"]);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTrace1 = scatter(x = vRGrid, y = vR, mode = "lines", text = "Log Likelihood", name = "Log Likelihood",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = [vRGrid[argmax(vR)]], y = [maximum(vR)], 
                  mode = "markers", text = "Maximum Value", name = "Maximum Value",
                  marker = attr(size = 12, color = "r"));

oLayout = Layout(title = "The Log Likelihood Function", width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "R", yaxis_title = "L(z; R)");
 hP = plot([oTrace1, oTrace2], oLayout);
 display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end


