# Test Code for Signal Processing
# Several tests for the Julia Code based on MATLAB Reference.
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
# - 1.0.000     26/10/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using BenchmarkTools;
using MAT;
using PlotlyJS;
# using UnicodePlots;

## Constants & Configuration

## External
include("JuliaInit.jl");
include("JuliaSignalProcessing.jl");

## General Parameters

figureIdx = 0;

exportFigures = false;


## Functions


## Parameters

# Data
matFileName = "TestSignalProcessing.mat";

dPadMode = Dict("circular" => PAD_MODE_CIRCULAR, "constant" => PAD_MODE_CONSTANT, "replicate" => PAD_MODE_REPLICATE, "symmetric" => PAD_MODE_SYMMETRIC);
dConvMode = Dict("full" => CONV_MODE_FULL, "same" => CONV_MODE_SAME, "valid" => CONV_MODE_VALID);


## Load / Generate Data

# dMatData = matread(matFileName);


## Analysis

# Pad Array

# Convolution 1D

# BEADS Filter

dMatData = matread("BEADS.mat");

vY          = vec(dMatData["vY"]);
modelDeg    = Int(dMatData["modelDeg"]);
fₛ          = dMatData["fs"];
asyRatio    = dMatData["asyRatio"];
λ₀          = dMatData["lam0"];
λ₁          = dMatData["lam1"];
λ₂          = dMatData["lam2"];
numIter     = 31;

vX, vF, vC = BeadsFilter(vY, modelDeg, fₛ, asyRatio, λ₀, λ₁, λ₂, numIter);

println(maximum(abs.(vX - vec(dMatData["vX"]))))
println(maximum(abs.(vF - vec(dMatData["vF"]))))


oTr1 = scatter(; y = vX, mode = "markers", name = "Julia");
oTr2 = scatter(; y = vec(dMatData["vX"]), mode = "markers", name = "MATLAB");
oLayout = Layout(title = "Julia vs. MATLAB", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

oTr1 = scatter(; y = vX, mode = "lines", name = "Julia");
oTr2 = scatter(; y = vec(dMatData["vX"]), mode = "lines", name = "MATLAB");
oLayout = Layout(title = "Julia vs. MATLAB", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2], oLayout);
display(hP);


oTr = scatter(; y = abs.(vX - vec(dMatData["vX"])), mode = "lines", name = "Julia");
oLayout = Layout(title = "Julia vs. MATLAB", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

oTr = scatter(; y = abs.(vX - vec(dMatData["vX"])) ./ abs.(vec(dMatData["vX"])), mode = "lines", name = "Julia");
oLayout = Layout(title = "Julia vs. MATLAB", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

oTr1 = scatter(; y = vF, mode = "lines", name = "Julia");
oTr2 = scatter(; y = vec(dMatData["vF"]), mode = "lines", name = "MATLAB");
oLayout = Layout(title = "Julia vs. MATLAB", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr1, oTr2], oLayout);
display(hP);

oTr = scatter(; y = vC, mode = "lines", name = "Julia");
oLayout = Layout(title = "Julia Cost", width = 600, height = 400, 
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
hP = Plot([oTr], oLayout);
display(hP);

## Display Results

