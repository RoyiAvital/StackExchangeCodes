# StackExchange Signal Processing Q29744
# https://dsp.stackexchange.com/questions/29744
# Amplitude and Phase Recovery of a Signal Embedded in Linear Signal with Noise
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
using Printf;
# External
using DelimitedFiles;
using FileIO;
import FreeType;
using LinearSegmentation; #<! See https://discourse.julialang.org/t/101248 and https://github.com/stelmo/LinearSegmentation.jl
using UnicodePlots;

## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions


## Parameters

# Data
vYFileName = "vY.csv";

# Model
minSegLen = 30.0;
maxRmse   = 0.05;

## Load / Generate Data

vY = readdlm(vYFileName, ',', Float64);
vX = 1:length(vY);


## Analysis

segs, fits = graph_segmentation(vX, vY; min_segment_length = minSegLen, max_rmse = 0.105);
# segs, fits = top_down(vX, vY; min_segment_length = minSegLen, max_rmse = 0.09);
# segs, fits = sliding_window(vX, vY; min_segment_length = minSegLen, max_rmse = 0.15);


# Remove the 1st item which is shared
for ii in 2:length(segs)
    deleteat!(segs[ii].idxs, 1);
end

## Display Results

# hC = BrailleCanvas(8, 90, origin_y = 0.0, origin_x = 0.0, height = 1.5, width = 510.0);
# hP = Plot(hC);
# hP = Plot([1], [1]);

# for (dataSeg, dataColor) in zip(segs, UnicodePlots.COLOR_CYCLE_FAINT)
#     # propertynames(segs[1]) #<! See https://stackoverflow.com/questions/41687418
#     # scatterplot!(hP, dataSeg.idxs, vY[dataSeg.idxs], color = dataColor);
#     # UnicodePlots.points!(hC, dataSeg.idxs, vY[dataSeg.idxs], color = dataColor);
#     # scatterplot!(hP, dataSeg.idxs, vY[dataSeg.idxs], color = dataColor);
#     if !(@isdefined hP)
#         global hP = scatterplot(dataSeg.idxs, vY[dataSeg.idxs]);
#     else
#         scatterplot!(hP, dataSeg.idxs, vY[dataSeg.idxs]);
#     end
# end

hP = scatterplot(segs[1].idxs, vY[segs[1].idxs], width = 90, height = 8, xlim = (vX[1], vX[end]), ylim = (minimum(vY), maximum(vY)));

for ii in 2:length(segs)
    scatterplot!(hP, segs[ii].idxs, vY[segs[ii].idxs]);
end

title!(hP, "Data");
xlabel!(hP, "Index");
ylabel!(hP, "Value");
display(hP);

maxSize = maximum([length(seg.idxs) for seg in segs]);
mW = Matrix{Int}(undef, maxSize, length(segs));
mW = zeros(Int, maxSize + 1, length(segs)); #<! So -1 will be in each column
mW[:] .= -1;
for ii in 1:length(segs)
    mW[1:length(segs[ii].idxs), ii] = segs[ii].idxs;
end

writedlm("mW.csv",  mW, ',');



