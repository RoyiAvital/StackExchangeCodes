# StackExchange Computational Science Q45061
# https://scicomp.stackexchange.com/questions/45061
# Covering Set of Scattered Points in R² with Discs with a Given Radius.
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     16/05/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using Distributions;
using HiGHS;
# using MAT;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
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

function RadiusConstrainedKMedoids( mX :: Matrix{T}, valR :: T ) where {T <: AbstractFloat}

    dataDim    = size(mX, 1);
    numSamples = size(mX, 2);

    # Distance Matrix: Squared Euclidean Distance
    # mD = (sum(abs2, mX; dims = 1) .+ sum(abs2, mX; dims = 1)) .- T(2.0) * (mX' * mX);
    mD = zeros(T, numSamples, numSamples);
    for ii ∈ 1:numSamples
        for jj ∈ 1:numSamples
            mD[ii, jj] = sum(abs2, mX[:, ii] - mX[:, jj]);
        end
    end

    # The value B_ij represents whether element i belongs to the cluster centered on element j
    mB = Variable(numSamples, numSamples, BinVar); #<! Indicator matrix

    # Constraints
    vConst = Constraint[mB[:, jj] <= mB[jj, jj] for jj ∈ 1:numSamples]; #<! Each sample belongs to one cluster, The center assigned to itself
    push!(vConst, sum(mB; dims = 2) == 1);                              #<! Each sample belongs to one cluster
    push!(vConst, maximum(mB .* mD) <= (valR * valR));                    #<! Limit the distance of each sample to its cluster center (Centroid)

    sConvProb = minimize( tr(mB), vConst );
    solve!(sConvProb, HiGHS.Optimizer; silent = true);

    mB = mB.value; #<! The value of the indicator matrix

    vSIdx = [ii for ii ∈ 1:numSamples if mB[ii, ii] > zero(T)];
    numClusters = length(vSIdx); #<! The number of clusters
    # Assign label per sample by its cluster center
    # vL = argmax(mB; dims = 2); #<! The label of each sample (Yields cartesian indices)
    vL = [argmax(vB) for vB in eachrow(mB)];
    # We could use `findfirst()` if `mB` is casted to `Bool`
    vI = unique(vL);
    # The centroid of the cluster as its mean
    mC = zeros(T, dataDim, numClusters); #<! The cluster centers
    # The cluster centers (Centroids - Mean of the samples assigned to the cluster)
    for ii ∈ 1:numClusters
        mC[:, ii] = mean(mX[:, vL .== vI[ii]]; dims = 2); #<! The cluster center
    end

    return mC, vL, vSIdx, mB;

end

## Parameters

# Data
numSamples = 120;

# Model
maxRadius = 0.5;


#%% Load / Generate Data

# Generate data as a mixture of distributions
vD = [Uniform(-1.0, 1.0), Normal(0.0, 0.6), Logistic(0.0, 0.2), Laplace(0.0, 0.4)];
mX = [rand(oRng, oD, 2, numSamples ÷ 4) for oD ∈ vD];
mX = hcat(mX...);

mC, vL, vSIdx, _ = RadiusConstrainedKMedoids(mX, maxRadius);
vI = unique(vL);
for ii ∈ 1:length(vI)
    vL[vL .== vI[ii]] .= ii;
end
mC = mX[:, vSIdx]; #<! K-Medoid Centroids


## Analysis


## Display Results

figureIdx += 1;

sTr = scatter(x = mX[1, :], y = mX[2, :], mode = "markers", name = "Data Samples");

oLayout = Layout(title = "Data Samples", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "x", yaxis_title = "y", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975));

vTr = [sTr];
hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end

figureIdx += 1;

sTr1 = scatter(x = mX[1, :], y = mX[2, :], marker_color = vL, marker_size = 8, mode = "markers", name = "Date Samples");
sTr2 = scatter(x = mC[1, :], y = mC[2, :], marker_size = 5, marker_symbol = "cross", mode = "markers", name = "Centroids");

vShp = [circle(x0 = vC[1] - maxRadius, y0 = vC[2] - maxRadius, 
                x1 = vC[1] + maxRadius, y1 = vC[2] + maxRadius;
                opacity = 0.15, fillcolor = "red", line_color = "red") for vC in eachcol(mC)];

oLayout = Layout(title = "Data Samples and Clusters", width = 600, height = 600, hovermode = "closest", 
                 xaxis_title = "x", yaxis_title = "y", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.025, y = 0.975), shapes = vShp);

vTr = [sTr1, sTr2];
hP = Plot(vTr, oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]); #<! https://github.com/JuliaPlots/PlotlyJS.jl/issues/491
end