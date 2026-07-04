# StackExchange Cross Validated Q496489
# https://stats.stackexchange.com/questions/496489
# Logistic Regression with Non Negative Constraints / Priors.
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
# - 1.0.000     01/03/2026  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function CVXSolver( mX :: Matrix{T}, vY :: Vector{N}, numCls :: N ) where {T <: AbstractFloat, N <: Integer}
    # Multinomial Logistic Regression with Non Negative Constraints / Priors.

    numSamples = size(mX, 1); #<! Number of samples
    dataDim    = size(mX, 2); #<! Number of features

    mW = Convex.Variable(numCls, dataDim);
    vB = Convex.Variable(numCls);

    valLoss = zero(T);

    for ii in 1:numSamples
        valLoss += Convex.logsumexp(mW * mX[ii, :] + vB) - Convex.dot(mW[vY[ii], :], mX[ii, :]) - vB[vY[ii]];
    end
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    sConvProb = minimize(valLoss, [mW >= 0.0] );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    mW = mW.value;
    vB = vec(vB.value);
    
    return mW, vB;

end


## Parameters

# Data
csvFileUrl      = raw"https://www.kaggle.com/api/v1/datasets/download/uciml/glass"; #<! See https://www.kaggle.com/datasets/uciml/glass
archiveFileName = "glass.zip";
csvFileName     = "Glass.csv";


## Load / Generate Data

if !isfile(csvFileName)
    if !isfile(archiveFileName)
        download(csvFileUrl, archiveFileName);
    end
    run(`powershell.exe -Command "Expand-Archive -Path '$archiveFileName' -DestinationPath '.' -Force"`)
end

# Load the CSV data
mData = readdlm(csvFileName, ',', Float64; skipstart = 1); #<! Data is a vector
mX = mData[:, 1:(end - 1)]; #<! Features
vY = Int.(mData[:, end]);   #<! Labels

numSamples = size(mX, 1);
dataDim    = size(mX, 2);

dCls = Dict{Int, String}(1 => "building_windows_float_processed", 2 => "building_windows_non_float_processed",
                      3 => "vehicle_windows_float_processed", 4 => "vehicle_windows_non_float_processed",
                      5 => "containers", 6 => "tableware", 7 => "headlamps");
numCls = length(dCls);


## Analysis

mW, vB = CVXSolver(mX, vY, numCls);
mP = mX * mW' .+ vB';
vŶ = [argmax(mP[ii, :]) for ii in 1:numSamples];

valAcc = (sum(vŶ .== vY) / numSamples) * 100.0;
@printf("Accuracy: %0.2f%%\n", valAcc);


## Display Results

# figureIdx += 1;

# vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# for (ii, methodName) in enumerate(keys(dSolvers))
#     vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
#                mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
# end
# oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

# hP = Plot(vTr, oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end

# figureIdx += 1;

# for (ii, methodName) in enumerate(keys(dSolvers))
#     vTr[ii] = scatter(x = 1:numIterations, y = dSolvers[methodName], 
#                mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0))
# end
# oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = "Objective Value");

# hP = Plot(vTr, oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end