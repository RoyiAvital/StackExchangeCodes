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
using DelimitedFiles;
using FileIO;
import FreeType;
using LinearAlgebra;
if Sys.iswindows()
    using MKL;
end
if Sys.isapple()
    using AppleAccelerate;
end
using NPZ;
using UnicodePlots;

## Constants & Configuration

## External
juliaInitPath = joinpath(".", "..", "..", "JuliaCode", "JuliaInit.jl")
include(juliaInitPath)

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions




## Parameters

# Data
# Repositories of the Olivetti Dataset
# https://github.com/essanhaji/face_recognition_pca
# https://github.com/jakeoeding/eigenfaces
olivettiFacesDataUrl    = raw"https://github.com/essanhaji/face_recognition_pca/raw/master/data/olivetti_faces.npy";
olivettiFacesTargetUrl  = raw"https://github.com/essanhaji/face_recognition_pca/raw/master/data/olivetti_faces_target.npy";

# Model
personThr   = 0.01;
faceThr     = 0.05;

## Load / Generate Data

mData = npzread(download(olivettiFacesDataUrl));

mX = collect(reshape(mData, 400, :)');
vY = npzread(download(olivettiFacesTargetUrl));
