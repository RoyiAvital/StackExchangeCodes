# LinearSegmentation.jl Test Case
# A test case of LinearSegmentation.jl.
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
# - 1.0.000     09/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Statistics;

# External
using DelimitedFiles;
using LinearSegmentation;
using StableRNGs;
using UnicodePlots;

## Constants & Configuration

oRng = StableRNG(123);

## Functions

function Conv1D( vA :: Vector{T}, vB :: Vector{T}; convMode :: String = "full" ) :: Vector{T} where {T <: Real}

    lenA = length(vA);
    lenB = length(vB);

    if (convMode == "full")
        startIdx    = 1;
        endIdx      = lenA + lenB - 1;
    elseif (convMode == "same")
        startIdx    = 1 + floor(Int, lenB / 2);
        endIdx      = startIdx + lenA - 1;
    elseif (convMode == "valid")
        startIdx    = lenB;
        endIdx      = lenA;
    end

    vO = zeros(T, lenA + lenB - 1);

    for idxB in 1:lenB
        @simd for idxA in 1:lenA
            @inbounds vO[idxA + idxB - 1] += vA[idxA] * vB[idxB];
        end
    end

    return vO[startIdx:endIdx];
end

function SolveMinCostPartitionIntervals( mD :: Matrix{T}, maxPartitions :: S  ) where {T, S <: Integer}

    numSamples = size(mD, 1);
    maxPartitions = min(maxPartitions, numSamples);
    mS = maximum(mD) .* ones(T, maxPartitions, numSamples); #<! Cost per segment
    mP = zeros(Int, maxPartitions, numSamples); #<! Path matrix

    mS[1, :] .= mD[1, :];
    mP[1, :] .= 1;
    for ii ∈ 2:maxPartitions
        for jj ∈ (ii + 1):numSamples
            minCost = 1e6;
            kkMin = 1;
            for kk ∈ 1:(ii - 1)
                currCost = mS[kk, ii - 1] + mD[ii, jj];
                if (currCost < minCost)
                    kkMin   = kk;
                    minCost = currCost;
                end
            mS[ii, jj] = minCost;
            mP[ii, jj] = kkMin;
            end
        end
    end

    return mS, mP;

end

function ExtractPath( mS :: Matrix{T}, mP :: Matrix{S} ) where {T, S <: Integer}

    numRows, numCols    = size(mS);
    vS                  = Vector{Vector{Int}}(undef, 0);

    startIdx = argmin(mS[:, numCols]);
    endIdx   = numCols;

    while ((startIdx > 0) && (endIdx > 0))
        prepend!(vS, [[startIdx, endIdx]]);
        colIdx      = mP[startIdx, endIdx];
        endIdx      = startIdx - 1;
        startIdx    = colIdx;
    end

    return vS;

end

function CalcDistMat( vX :: Vector{T} ) where {T}
    
    # numSamples  = length(vX);
    mD = ((vX .- vX') .^ 2);

    return mD;

end

function PolyFit( vX :: AbstractVector{T}, vY :: AbstractVector{T}, polyDeg :: Int ) where {T}

    numSamples = length(vX);
    mM = zeros(T, numSamples, polyDeg + 1);

    for ii = 1:(polyDeg + 1)
        mM[:, ii] = vX .^ (ii - 1);
    end

    return (mM \ vY);

end

function PolyVal( vX :: AbstractVector{T}, vP :: Vector{S}, polyDeg :: Int ) where {T, S}

    numSamples = length(vX);
    mM = zeros(T, numSamples, polyDeg + 1);

    for ii = 1:(polyDeg + 1)
        mM[:, ii] = vX .^ (ii - 1);
    end

    return mM * vP;

end


function CalcDistMatReg(vX :: Vector{T}, vY :: Vector{T}, hLossFun :: Function; minLen :: S = 0.0, maxLen :: S = 1.0, maxLoss :: S = 0.9, maxDist :: S = inf) where {T, S}
    # TODO: Use symmetric matrix

    numSamples = length(vX);
    mD = zeros(numSamples, numSamples);

    for ii in 1:numSamples, jj in ii:numSamples
        if (abs(vX[ii] - vX[jj]) < minLen)
            mD[ii, jj] = maxDist;
            mD[jj, ii] = maxDist;
            continue;
        end

        if (abs(vX[ii] - vX[jj]) > maxLen)
            mD[ii, jj] = maxDist;
            mD[jj, ii] = maxDist;
            continue;
        end

        @views vP = PolyFit(vX[ii:jj], vY[ii:jj], 1);
        @views vE = PolyVal(vX[ii:jj], vP, 1);
        # estMse = mean(abs2, vE - @view vY[ii:jj]);
        @views estLoss = hLossFun(vY[ii:jj], vE);

        if (estLoss > maxLoss)
            mD[ii, jj] = maxDist;
            mD[jj, ii] = maxDist;
        else
            mD[ii, jj] = estLoss;
            mD[jj, ii] = estLoss;
        end
    end

    return mD;

end

# Loss Fun -> Minimize (Like Distance)
hLossFunMse(vY, vYY) = mean(abs2, vY - vYY); #<! vY Ground Truth, vYY - Estimation
# AffinityFun -> Maximize (Like Affinity)
hAffFunR2(vY, vYY) = 1 - (sum(abs2, vY .- vYY) / sum(abs2, mean(vY) .- vYY)); #<! vY Ground Truth, vYY - Estimation
hLossFunR2(vY, vYY) = -hAffFunR2(vY, vYY);



# vX = [1, 1.1, 0.9, 7, 8, 7.5, 4, 3.6, 4.4];
# mD = CalcDistMat(vX)
# mS, mP = SolveMinCostPartitionIntervals(mD, 1000)

# writedlm("mS.csv",  mS, ',');
# writedlm("mP.csv",  mP, ',');


## Parameters

# Data
ampFiltSize   = 20;
phaseFiltSize = 50;

vSeg = [1, 151, 301, 401, 501];
vSeg = [1, 11, 21, 31, 41];

numSamples = vSeg[end] - 1;

# Model
minSegLen = 5.0;
maxSegLen = 1000.0;
maxRmse   = 1.75;

## Load / Generate Data

vAmp    = rand(oRng, numSamples);
vAmp    = 0.2 * Conv1D(vAmp, ones(ampFiltSize) / ampFiltSize; convMode = "same");
vPhase  = 0.2 * rand(oRng, numSamples);
vPhase  = Conv1D(vPhase, ones(phaseFiltSize) / phaseFiltSize; convMode = "same");
vPhase  = cumsum(vPhase);

vX = LinRange(0, numSamples - 1, numSamples);

vC = vAmp .* cos.(2 * pi * vPhase);
vL = zeros(numSamples);
vL[vSeg[1]:(vSeg[2] - 1)] .= 0;
vL[vSeg[2]:(vSeg[3] - 1)] .= 1;
vL[vSeg[3]:(vSeg[4] - 1)] .= collect(LinRange(0.5, 1.0, vSeg[4] - vSeg[3]));
vL[vSeg[4]:(vSeg[5] - 1)] .= collect(LinRange(1.0, 0.4, vSeg[5] - vSeg[4]));

vY = vC .+ vL;

## Display Data

ii = 1;
vIdx = vSeg[ii]:(vSeg[ii + 1] - 1)

hP = scatterplot(vX[vIdx], vY[vIdx], width = 90, height = 8, xlim = (vX[1], vX[end]), ylim = (minimum(vY), maximum(vY)));

for ii in 2:(length(vSeg) - 1)
    local vIdx = vSeg[ii]:(vSeg[ii + 1] - 1)
    scatterplot!(hP, vX[vIdx], vY[vIdx]);
end

title!(hP, "Input Data");
xlabel!(hP, "Index");
ylabel!(hP, "Value");
display(hP);


## Analysis

# LinearSegmentation.jl
# segs = shortest_path(vX, vY; min_segment_length = minSegLen, overlap = true, fit_function = :rmse, fit_threshold = maxRmse);


# # Remove the 1st item which is shared
# for ii in 2:length(segs)
#     deleteat!(segs[ii][1], 1);
# end

# ## Display Results

# hP = scatterplot(segs[1][1], vY[segs[1][1]], width = 90, height = 8, xlim = (vX[1], vX[end]), ylim = (minimum(vY), maximum(vY)));

# for ii in 2:length(segs)
#     scatterplot!(hP, segs[ii][1], vY[segs[ii][1]]);
# end

# title!(hP, "Linear Segmentation");
# xlabel!(hP, "Index");
# ylabel!(hP, "Value");
# display(hP);


# for ii in 1:length(segs)
#     println("The $(ii) segment is $(segs[ii][1][1]):$(segs[ii][1][end])");
# end

# My own solution
vX = collect(vX);
# mD = CalcDistMatReg(vX, vY; minLen = minSegLen, maxLen = maxSegLen, maxMSE = maxRmse * maxRmse, maxDist = 1e6);
mD = CalcDistMatReg(vX, vY, hLossFunMse; minLen = minSegLen, maxLen = maxSegLen, maxLoss = maxRmse * maxRmse, maxDist = 1e6);
mSS, mPP = SolveMinCostPartitionIntervals(mD, 1000);
vPP = ExtractPath(mSS, mPP);



