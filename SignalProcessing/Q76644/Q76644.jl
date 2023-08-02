# StackExchange Signal Processing Q76644
# https://dsp.stackexchange.com/questions/76644
# Estimating the Frequency of a Sinusoidal Signal in White Noise
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
# - 1.0.000     15/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal
# using LinearAlgebra;
using Printf;
using Statistics;
# External
using CairoMakie;
using FFTW;


## Constants & Configuration

## External
juliaInitPath = joinpath(".", "..", "..", "JuliaCode", "JuliaInit.jl")
include(juliaInitPath)

## General Parameters

figureIdx       = 0;
exportFigures   = true;

## Functions

function EstPeakShift( vP :: AbstractVector{T} ) where {T <: AbstractFloat}
    # Estimates normalized shift to the peak using Quadratic Model.
    # Assumed -1 -> vP[1], 0 -> vP[2], 1 -> vP[3].
    # Mutate vP.

    vP .= vP .- vP[2];

    # Basically (-b / 2a)
    normalizedShift = (vP[1] - vP[3]) / (2 * (vP[1] + vP[3]));

    return normalizedShift;

end

function EstDataFreq( vX :: Vector{T}, vB :: Vector{T}, samplingFreq :: T; paddingFactor :: S = 3 ) where {T <: AbstractFloat, S <: Integer}

    numSamples  = length(vX);
    numSamplesDft = ceil(Int, 2 ^ (log2(numSamples) + paddingFactor));
    vXP = append!(copy(vX), zeros(numSamplesDft - numSamples));
    vXX  = rfft(vXP);
    _, idxK = findmax(abs2, vXX[2:(end - 1)]);
    idxK += 1; #<! Compensate for the view
    vB .= abs.(vXX[(idxK - 1):(idxK + 1)]); #<! Using a buffer, Check if `abs()` is better
    freqShift = EstPeakShift(vB);

    return (samplingFreq / numSamplesDft) * (idxK - 1 + freqShift);

end


function EstDataFreqCedron( vX :: Vector{T}, samplingFreq :: T ) where {T <: AbstractFloat}
    # Exact Frequency Formula for a Pure Real Tone
    # Cedron Dawg
    # https://www.dsprelated.com/showarticle/773.php

    N = length(vX);
    vXX = rfft(vX);
    ~, idxK = findmax(abs2, @view vXX[2:(end - 1)]);
    idxK += 1;
    vXK = @view vXX[(idxK - 1):(idxK + 1)];
    r = exp(-1im * 2π / N);
    vCosB = cos.(2π / N .* [idxK - 2, idxK - 1, idxK]); #<! Zero based like DFT
    num = (-vXK[1] * vCosB[1]) + (vXK[2] * (1 + r) * vCosB[2]) - (vXK[3] * r * vCosB[3]);
    den = -vXK[1] + (vXK[2] * (1 + r)) - vXK[3] * r;
    f = real(acos(num / den)) / (2π);

    return f;

end

function EstimateSineFreqCedron3Bin( vX :: Vector{T}, samplingFreq :: T ) where {T <: AbstractFloat}
    # Improved Three Bin Exact Frequency Formula for a Pure Real Tone in a DFT
    # Cedron Dawg
    # https://www.dsprelated.com/showarticle/1108.php
    
    numSamples = length(vX);
    
    vXK = rfft(vX);
    
    ~, idxK = findmax(abs2, @view vXK[2:(end - 1)]);
    idxK += 1;
    
    vZ = @view vXK[(idxK - 1):(idxK + 1)];
    
    vR = real.(vZ);
    vI = imag.(vZ);
    iRoot2 = 1 / sqrt(2);
    vBetas = [idxK - 2, idxK - 1, idxK] * (2π / numSamples); #<! Zero based like DFT
    vCosB = cos.(vBetas);
    vSinB = sin.(vBetas);
    
    vA = [iRoot2 * (vR[2] - vR[1]); iRoot2 * (vR[2] - vR[3]); vI[1]; vI[2]; vI[3]];
    vB = [iRoot2 * (vCosB[2] * vR[2] - vCosB[1] * vR[1]); iRoot2 * (vCosB[2] * vR[2] - vCosB[3] * vR[3]); vCosB[1] * vI[1]; vCosB[2] * vI[2]; vCosB[3] * vI[3]];
    vC = [iRoot2 * (vCosB[2] - vCosB[1]); iRoot2 * (vCosB[2] - vCosB[3]); vSinB[1]; vSinB[2]; vSinB[3]];
    
    normC = sqrt(sum(abs2, vC));
    vP = vC / normC;
    vD = vA .+ vB;
    vK = vD .- (vD' * vP) * vP;
    num = vK' * vB;
    den = vK' * vA;
    
    ratio = max(min(num / den, 1), -1);
    estFreq = acos(ratio) / (2π) * samplingFreq;

    return estFreq;
    
    
end


function EstDataFreqKay( vX :: Vector{T}, samplingFreq :: T ) where {T <: AbstractFloat}

    numSamples = length(vX);
    weightDen  = (numSamples ^ 3) - numSamples;
    estFreq    = 0;
    for ii in 1:(numSamples - 1)
        sampleWeight = (6 * ii * (numSamples - ii)) / weightDen;
        estFreq     += (sampleWeight * angle(vX[ii] * vX[ii + 1]));
    end
    return (estFreq / (2π)) * samplingFreq;

end

## Parameters

# Signal Parameters
numSamples      = 100;
samplingFreq    = 1.0; #<! The CRLB is for Normalized Frequency

# Sine Signal Parameters (Non integers divisors of N requires much more realizations).
# sineFreq    = 0.24; #<! Do for [0.05, 0.10, 0.25] For no integer use 0.37.
sineAmp     = 10; #<! High value to allow high SNR
vT          = 0:(numSamples - 1);

# Buffers
vB = zeros(3); #<! 3 samples (Around the peak)
vS = zeros(numSamples);
vW = zeros(numSamples);
vX = zeros(numSamples);

# Analysis Parameters
numRealizations = 750;
# SNR of the Analysis (dB)
vSnrdB = LinRange(-10, 50, 150);
# vSnrdB = LinRange(30, 50, 150);

## Load / Generate Data

numNoiseStd = length(vSnrdB);
vNoiseStd   = zeros(numNoiseStd);

for ii = 1:numNoiseStd
    vNoiseStd[ii] = sqrt((sineAmp * sineAmp) / (2 * 10 ^ (vSnrdB[ii] / 10))); 
end

mFreqErr = zeros(numRealizations, numNoiseStd);

## Analysis

# sineFreq = 0.4 * rand();
# angFreq = 2π * (sineFreq / samplingFreq); #<! Make sure (sineFreq / samplingFreq < 0.5)
# sinePhase = 2π * rand();
# vS = sineAmp .* sin.((angFreq .* vT) .+ sinePhase);

# println(EstDataFreq(vS, vB, samplingFreq))
# println(sineFreq)

for jj in 1:numNoiseStd, ii in 1:numRealizations
    sineFreq = 0.05 + (0.35 * rand());
    sineFreq = 0.25;
    angFreq = 2π * (sineFreq / samplingFreq); #<! Make sure (sineFreq / samplingFreq < 0.5)
    sinePhase = 2π * rand();
    vS .= sineAmp .* sin.((angFreq .* vT) .+ sinePhase);
    vW .= vNoiseStd[jj] .* randn(numSamples);
    vX .= vS .+ vW;
    # mFreqErr[ii, jj] = sineFreq - EstDataFreq(vX, vB, samplingFreq);
    # mFreqErr[ii, jj] = sineFreq - EstDataFreqCedron(vX, samplingFreq);
    # mFreqErr[ii, jj] = sineFreq - EstimateSineFreqCedron3Bin(vX, samplingFreq);
    mFreqErr[ii, jj] = sineFreq - EstDataFreqKay(vX, samplingFreq);
end

vFreqErr = mean(mFreqErr .^ 2; dims = 1);
vFreqErr = vFreqErr[:];

sineMse     = (sineAmp * sineAmp) / 2;
vNoiseVar   = vNoiseStd .^ 2;
vSnr        = sineMse ./ vNoiseVar;

# CRLB
# 12 -> Sine / Cosine
# 6 -> Exp
vFreqMseCrlb = (12 * samplingFreq * samplingFreq) ./ (((2π) ^ 2) .* vSnr .* ((numSamples ^ 3) - numSamples));





## Display Results

figureIdx += 1;

hF = Figure(resolution = (800, 800));
hA = Axis(hF[1, 1], title = "Frequency Estimation", xlabel = "SNR [dB]", ylabel = "MSE [dB]");
lines!(vSnrdB, 10 .* log10.(vFreqMseCrlb), linewidth = 3, label = "CRLB");
lines!(vSnrdB, 10 .* log10.(vFreqErr), linewidth = 3, label = "Quadratic Interpolation");
axislegend();
display(hF);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end
