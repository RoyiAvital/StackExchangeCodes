# StackExchange Signal Processing Q96559
# https://dsp.stackexchange.com/questions/96559
# Utilize the DFT For Least Squares Estimation of a Single Tone Parameters to Approximate Arbitrary Signal.
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
# - 1.0.000     28/04/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;     #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FFTW;
using FindPeaks1D;
using Optim;
# using Peaks;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl")); #<! Display Images

## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function GenModelMatrix( valF :: T, numSamples :: N ) where {T <: AbstractFloat, N <: Integer}
    
    mS = zeros(T, numSamples, 2);
    for nn ∈ 1:numSamples
        mS[nn, 1] = cospi(T(2) * valF * nn);
        mS[nn, 2] = sinpi(T(2) * valF * nn);
    end

    return mS;

end

function CalcMse( vX :: Vector{T}, valF :: T ) where {T <: AbstractFloat}
    
    numSamples = length(vX);
    mS         = GenModelMatrix(valF, numSamples);   

    vW     = mS \ vX;
    valMse = mean(abs2, mS * vW - vX);

    return valMse;

end

## Parameters

fileName = "Signal.csv" #<! From https://gist.github.com/cdboschen/cb9d1cb6976101e22d0295c5c16b64c8 (By @Dan Boschen)

# Problem parameters
samplingFreq  = 10e6;
peakRadius    = 10_000;
peakMinHeight = 1_000;
optRadius     = 10.0 / samplingFreq;

numGridPts = 5_000;

## Load / Generate Data

# Load the Signal
# Bandlimited noise waveform extending from 1 to 3 MHz.
# 3 Tones:
# - peak = 2.2, freq = 501 kHz, phase = 0.35 radians.
# - peak = 1.6, freq = 2.1 MHz, phase = 0.1 radians.
# - peak = 0.5, freq = 4.1 MHz, phase = -2.1 radians.
# Sampling Rate at 10 MHz.
vX = vec(readdlm(fileName));

## Analysis

numSamples = length(vX);

hF(valF :: T) where {T <: AbstractFloat} = CalcMse(vX, valF);

# Local Peaks

vXDft       = rfft(vX);
vFFreq      = rfftfreq(numSamples, samplingFreq);
vPeakIdx, _ = findpeaks1d(abs.(vXDft); distance = peakRadius, height = peakMinHeight);

# Loss Function on a Grid
vF = [collect(LinRange(0, 0.5, numGridPts)); vFFreq[vPeakIdx] ./ samplingFreq];
vF = sort(vF);
vL = [hF(valF) for valF in vF];

# Local Optimization
numPeaks = length(vPeakIdx);
mP = zeros(3, numPeaks); #<! (Initial Frequency, Estimated Frequency, Loss at Estimated Frequency)

for ii ∈ 1:numPeaks
    valF = vFFreq[vPeakIdx[ii]] / samplingFreq;
    sOpt = optimize(hF, valF - optRadius, valF + optRadius);
    estF = sOpt.minimizer;
    losF = sOpt.minimum;
    
    mP[1, ii] = valF;
    mP[2, ii] = estF;
    mP[3, ii] = losF;
end

# mP = mP[:, sortperm(mP[3, :])];

## Display Results

# Display Data
figureIdx += 1;

oTr     = scatter(; x = vFFreq, y = abs.(vXDft), mode = "lines", 
                  line = attr(width = 2.0),
                  text = "DFT", name = "DFT");
oLayout = Layout(title = "The Data DFT", width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "Frequency [Hz]", yaxis_title = "Amplitude",
                  margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot(oTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

figureIdx += 1;

# Plotly Marker Symbols
# https://stackoverflow.com/questions/65946833
# https://plotly.com/python/marker-style
vMarkerSymbol = ["circle", "circle-dot", "square", "square-dot", "diamond", "diamond-dot", "cross", "cross-dot", "x", "x-dot", "pentagon", "pentagon-dot", "star", "star-dot", "hexagram", "hexagram-dot", "traingle-up", "traingle-up-dot", "triangle-down", "triangle-down-dot"];
vMarkerSymbol = vMarkerSymbol[1:2:end];

numPeaks      = min(numPeaks, 10);
vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, 2 * numPeaks + 1);

vTr[1]  = scatter(; x = vF * samplingFreq, y = vL, mode = "lines", 
                  line = attr(width = 2.0),
                  text = "Loss", name = "Loss");

jj = 1;
kk = 0;
for ii ∈ 1:numPeaks
    global jj += 1;
    global kk += 1;
    vTr[jj] = scatter(; x = [mP[1, ii] * samplingFreq], y = [hF(mP[1, ii])], mode = "markers",
                      marker = attr(size = 15, color = vPlotlyDefColors[kk], symbol = vMarkerSymbol[kk]),
                      text = "initialization $(ii)", name = "initialization $(ii)");
    jj += 1;
    vTr[jj] = scatter(; x = [mP[2, ii] * samplingFreq], y = [hF(mP[2, ii])], mode = "markers",
                      marker = attr(size = 15, color = vPlotlyDefColors[kk], symbol = vMarkerSymbol[kk] * "-dot"),
                      text = "Optimized $(ii)", name = "Optimized $(ii)");
end
oLayout = Layout(title = "The Loss Function and Optimization", width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "Frequency [Hz]", yaxis_title = "MSE",
                  margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end
