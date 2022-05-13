# StackExchange Signal Processing Q82918
# https://dsp.stackexchange.com/questions/82918
# MUSIC Algorithm for Direction of Arrival (DOA) in Acoustic Signals
# References:
#   1.  
# Remarks:
#   1.  B
# TODO:
# 	1.  C
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     09/05/2022
#   *   First release.

## Packages

using LinearAlgebra;
using DSP;
using Plots;
using Printf;

## Constants

const SPEED_OF_LIGHT_M_S = 3e9;

## General Parameters

figureIdx = 0;
# gr();
plotlyjs(); #<! Seems to handle the data 
# inspectdr();

## Functions

function CalcPhaseVec!( vP :: Vector{T}, aziAngle :: T, distElm :: S, sigFreq :: S, numElements :: Integer ) where{T <: AbstractFloat, S <: Number}

    constVal = (distElm * sind(aziAngle) / SPEED_OF_LIGHT_M_S) * sigFreq;

    for (ii, iVal) in enumerate(0:(numElements - 1))
        vP[ii] = constVal * iVal;
    end

end

function CalcPhaseElement( aziAngle :: T, elmntIdx :: Integer ) where{T <: AbstractFloat}

    return sind(aziAngle) * CONST_PHASE_FCTR * elmntIdx;

end

function CalcSteeringMatrix!( mA :: Matrix{T}, numSig :: Integer, vSigAzimuth :: Vector{S}, distElm :: S, sigFreq :: S, numElements :: Integer ) where{T <: Complex{<: AbstractFloat}, S <: Number}
    
    for jj in 1:numSig
        for ii in 1:numElements
            mA[ii, jj] = exp(-2im * π * CalcPhaseElement(vSigAzimuth[jj], ii - 1));
        end
    end

end

function CalcCovMat( mA :: Matrix{T} ) where{T <: Complex} #<! Complex{<: AbstractFloat} is equivalent of Complex

    numRows, numCols = size(mA);

    mC = Matrix{T}(undef, numRows, numCols);

    for jj in 1:numCols
        colMean = sum(@view mA[:, jj]) / numRows;
        for ii in 1:numRows
            mC[ii, jj] = mA[ii, jj] - colMean;
        end
    end

    mC = mC' * mC;
    mC ./ (numRows - 1);

end

## Parameters

# Array
numElements = 20;
distElmFctr = 1; #<! Factor of element distance (Units fo Lambda / 2)

# Signals
timeInterval    = 5.0;
vSigAmp         = [1.0, 1.0, 1.0];
sigFreq         = 1e1; #<! [Hz]
vSigPhase       = [0.0, 29.0, 61.0]; #<! [Deg]
vSigAzimuth     = [-60.0, 0.0, 40.0];
samplingFreq    = 10 * sigFreq; #<! [Hz]

# Noise
noiseAmp = 0.01; #<! Standard deviation

# MUSIC
numGridPts = 361; #<! Grid of Angles

waveLen = SPEED_OF_LIGHT_M_S / sigFreq;
distElm = distElmFctr * (waveLen / 2.0);

const CONST_PHASE_FCTR = (distElm / SPEED_OF_LIGHT_M_S) * sigFreq;


## Generate / Load Data

numSamples  = round(Integer, samplingFreq * timeInterval);
numSig      = length(vSigAmp);
vT = LinRange(0.0, timeInterval, numSamples + 1)[1:numSamples];

mR = Matrix{Float64}(undef, numSamples, numSig);

for jj = 1:numSig
    sigPhase = vSigPhase[jj] / (2π); #<! [Radians]
    mR[:, jj] .= vSigAmp[jj] .* sin.(2π .* sigFreq .* vT .+ sigPhase);
end

## MUSIC Algorithm

mA = Matrix{ComplexF64}(undef, numElements, numSig);
CalcSteeringMatrix!(mA, numSig, vSigAzimuth, distElm, sigFreq, numElements);

mX = hilbert(mR) * mA';
mX .+= noiseAmp .* randn(ComplexF64, numSamples, numElements);

mC = CalcCovMat(mX);

# Julia's Eigen Decomposition is different from MATLAB's.
# Hence the results are different from MATLAB's code.
mE = eigvecs(mC, sortby = x -> abs2(x)); #<! Eigen Vectors (Spanning space of Signal / Noise)
mV = mE[:, 1:(numElements - numSig)]; #<! Spanning the Signal (Assuming Signal Eigen Values are larger)

vTheta = LinRange(-90, 90, numGridPts); #<! Grid for Estimation
vS = Vector{ComplexF64}(undef, numElements);
vB = Vector{ComplexF64}(undef, numElements); #<! Buffer
vM = Vector{Float64}(undef, numGridPts);

mVV = Hermitian(mV * mV');
mU = sqrt(mVV);

for ii in 1:numGridPts
    for jj in 1:numElements
        vS[jj] = exp(-2im * π * CalcPhaseElement(vTheta[ii], jj - 1));
    end
    mul!(vB, mU, vS);
    vM[ii] = 1 / real(vB' * vB);
end


## Display Results

figureIdx += 1;
fileName = @sprintf "Figure%04d.png" figureIdx;
hP = plot(vTheta, 10 * log10.(vM));
display(hP);
# png(hP, fileName);