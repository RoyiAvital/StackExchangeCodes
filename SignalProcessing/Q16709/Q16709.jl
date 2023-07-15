# StackExchange Signal Processing Q16709
# https://dsp.stackexchange.com/questions/16709
# Face Recognition by Eigen Faces Algorithm
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
using Printf;
# External
using LinearAlgebra;
if Sys.iswindows()
    using MKL;
end
if Sys.isapple()
    using AppleAccelerate;
end
using NPZ;
using PlotlyJS;
using Statistics;

## Constants & Configuration

## External
juliaInitPath = joinpath(".", "..", "..", "JuliaCode", "JuliaInit.jl")
include(juliaInitPath)

## General Parameters

figureIdx       = 0;
exportFigures   = true;

## Functions


## Parameters

# Data
# Repositories of the Olivetti Dataset
# https://github.com/essanhaji/face_recognition_pca
# https://github.com/jakeoeding/eigenfaces
olivettiFacesDataUrl    = raw"https://github.com/essanhaji/face_recognition_pca/raw/master/data/olivetti_faces.npy";
olivettiFacesTargetUrl  = raw"https://github.com/essanhaji/face_recognition_pca/raw/master/data/olivetti_faces_target.npy";

dataXFileName = "mX.npy";
dataYFileName = "vY.npy";

# Model
numDims     = 16;
personThr   = 0.01;
faceThr     = 0.05;

## Load / Generate Data
if !(isfile(dataXFileName))
    download(olivettiFacesDataUrl, dataXFileName);
end
if !(isfile(dataYFileName))
    download(olivettiFacesTargetUrl, dataYFileName);
end

mX = npzread(dataXFileName);
mX = collect(reshape(mX, 400, :)'); #<! Each column is a 64x64 image
vY = npzread(dataYFileName);

dataDim   = size(mX, 1); #<! D
numImages = size(mX, 2); #<! N

## Analysis

# Center and Normalize Data
vμ = mean(mX; dims = 2); #<! Average image
mX .-= vμ;
mX ./= std(mX; mean = vμ, dims = 2); #<! We can use pre calculated mean

# Generate the Covariance Matrix
mC = (mX * mX') ./ numImages;

# Eigen Decomposition
mE = eigen(mC); #<! Sorted in ascending order
mU = mE.vectors[:, (end - numDims + 1):end];

# Encoding
mZ = mU' * mX;

vC = unique(vY);
numClass = length(vC);
mZμ = zeros(numDims, numClass);

for ii = 1:numClass
    mZμ[:, ii] = mean(mZ[:, vY .== vC[ii]]; dims = 2);
end

# Histogram of distance within class and inter class
# Not efficient
vHIn  = []; #<! Samples inside the class
vHOut = []; #<! Samples outside the class

for ii = 1:numClass
    global vHIn;
    global vHOut;
    append!(vHIn, sqrt.(sum(abs2, mZ[:, vY .== vC[ii]] .- mZμ[:, ii], dims = 1)));
    append!(vHOut, sqrt.(sum(abs2, mZ[:, vY .!= vC[ii]] .- mZμ[:, ii], dims = 1)));
end

## Display Results

figureIdx += 1;

oTrace = scatter(x = mZ[1, :], y = mZ[2, :], mode = "markers", 
text = string.(vY), marker = attr(size = 12, color = vY, showscale = true));
oLayout = Layout(title = "Encoded Data", width = 600, height = 600, hovermode = "closest");
hP = plot([oTrace], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTrace = scatter(x = mZμ[1, :], y = mZμ[2, :], mode = "markers", 
text = string.(vY), marker = attr(size = 12, color = vC, showscale = true));
oLayout = Layout(title = "Encoded Data Mean", width = 600, height = 600, hovermode = "closest");
hP = plot([oTrace], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

figureIdx += 1;

oTrace1 = histogram(x = vHIn, opacity = 0.75, histnorm = "probability", name = "Inside Class");
oTrace2 = histogram(x = vHOut, opacity = 0.75, histnorm = "probability", name = "Outside Class");
oLayout = Layout(title = "Distance Histogram, d = $(numDims)", width = 600, height = 600, barmode = "overlay", hovermode = "closest");
hP = plot([oTrace1, oTrace2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

