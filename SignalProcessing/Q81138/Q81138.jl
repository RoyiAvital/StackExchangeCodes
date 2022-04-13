# StackExchange Signal Processing Q81138
# https://dsp.stackexchange.com/questions/81138
# Using Least Mean Square (LMS) Filter for Beamforming on Linear Array in
# Julia
# References:
#   1.  
# Remarks:
#   1.  B
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     09/04/2022
#   *   First release.

## Packages

using LinearAlgebra;
using DSP;
using Plots;

## Constants

const SPEED_OF_LIGHT_M_S = 3e9;

## General Parameters

figureIdx = 0;
# gr();
plotlyjs();
# inspectdr();

## Functions

function CalcPhaseVec!( vP :: Vector{T}, aziAngle :: T, distElm :: S, sigFreq :: S, numElements :: Integer ) where{T <: AbstractFloat, S <: Number}

    constVal = (distElm * sind(aziAngle) / SPEED_OF_LIGHT_M_S) * sigFreq;

    for (ii, iVal) in enumerate(0:(numElements - 1))
        vP[ii] = constVal * iVal;
    end

end

function GenSensorsSignal!( mX :: Matrix{T}, vS :: Vector{T}, vP :: Vector{T} ) where{T <: AbstractFloat}
    
    vSS = hilbert(vS);
    for ii in 1:size(mX, 1)
        for jj in 1:size(mX, 2)
            mX[ii, jj] = real(vSS[ii] * exp(-(2im)π * vP[jj]))
        end
    end

end

function LmsFilter!( vW :: Vector{T}, mY :: Matrix{T}, vD :: Vector{T}, numSamples :: Integer, stepSize :: T; normalizeMode :: Bool = true ) where{T <: AbstractFloat}

    DELTA_PARAM = 1e-5;

    for ii in 1:numSamples
        vY = @view mY[ii, :];
        
        zSample = dot(vW, vY);
        eSample = vD[ii] - zSample;

        vW .+= ifelse(normalizeMode, stepSize .* eSample .* vY, (stepSize / (DELTA_PARAM + dot(vY, vY))) .* eSample .* vY);

    end

end

function CalcUlaPattern!( vH :: Vector{C}, vC :: Vector{C}, vW :: Vector{T}, vA :: AbstractVector{T}, distElm :: T, sigFreq :: T, numElements :: Integer ) where{C <: Complex, T <: AbstractFloat}
    # Using `AbstractVector{T}` allows supporting range vectors.

    @. vC = -(2im)π * ((distElm * sin(vA)) / SPEED_OF_LIGHT_M_S) * sigFreq;
    
    vH .= vW[1];
    
    for ii in 2:numElements
        vH .+= vW[ii] .* exp.(vC .* (ii - 1))
    end

end



## Parameters

# Array
numElements = 40;
distElmFctr = 1; #<! Fector of el

# Signals
timeInterval    = 5.0;
sigFreq         = 1e3; #<! [Hz]
samplingFreq    = 100 * sigFreq; #<! [Hz]

# Target
targetAmp       = 1;
targetAzimuth   = 30.0; #<! [Deg]

# Interference
vIntAmp     = [0; 0; 0];
vIntAzimuth = [20; -30; 50]; #<! [Deg]

# Noise
noiseAmp = 0.1; #<! Standard deviation

# LMS Filter
vW = zeros(numElements); #<! Initial Weight Vector
stepSize = 5e-4;

# Analysis
numAzimuths = 720;

## Generate / Load Data

numSamples = round(Integer, samplingFreq * timeInterval);
vT = LinRange(0.0, timeInterval, numSamples + 1)[1:numSamples];

waveLen = SPEED_OF_LIGHT_M_S / sigFreq;
distElm = distElmFctr * (waveLen / 2.0);


vR = sin.(2π .* sigFreq .* vT); #<! Reference signal

vP = zeros(numElements);


CalcPhaseVec!(vP, targetAzimuth, distElm, sigFreq, numElements);

mX = zeros(numSamples, numElements);

GenSensorsSignal!(mX, vR, vP);

# plot(mX, size = (1050, 700));

## LMS

vWW = copy(vW);

LmsFilter!(vWW, mX, vR, numSamples, stepSize);


## Analysis

recError = norm(mX * vWW - vR);
println("Reconstruction Error: $recError");

vH = zeros(ComplexF64, numAzimuths);
vA = LinRange(0, 2π, numAzimuths + 1)[1:numAzimuths];
vC = zeros(ComplexF64, numAzimuths);
CalcUlaPattern!(vH, vC, vWW, vA, distElm, sigFreq, numElements);

## Display Results

plot(vA, abs.(vH), proj = :polar, size = (1050, 700));