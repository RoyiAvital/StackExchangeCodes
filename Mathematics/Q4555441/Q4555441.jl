# StackExchange Mathematics Q4555441
# https://math.stackexchange.com/questions/4555441
# Estimate Gaussian Mixture Model (GMM) Parameters Embedded in Linear System.
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
# - 1.0.000     30/06/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using DelimitedFiles;
# using GaussianMixtures;
using LeastSquaresOptim;
using LsqFit;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
# include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);

## Functions

function ObjFun( mA, vT, vB, vX, vP )

    return mean(abs2, ResFun(mA, vT, vB, vX, vP));

end

function ResFun( mA, vT, vB, vX, vP )

    vY = ModelFun(mA, vT, vX, vP);

    return vY - vB; #<! TODO: Can be optimized to inplace

end

function ModelFun( mA, vT, vX, vP )

    vF = view(vP, 1:5);
    vμ = view(vP, 6:10);
    vW = view(vP, 11:15);

    fill!(vX, zero(eltype(mA)));

    for ii ∈ 1:5
        vX .+= vF[ii] .* exp.(- (Base.power_by_squaring.(vT .- vμ[ii], 2) ./ vW[ii]) );
        # vX .+= vF[ii] .* exp.(- (((vT .- vμ[ii]) .^ 2.0) ./ vW[ii]) );
    end

    return mA * vX; #<! TODO: Can be optimized to inplace

end

## Parameters

# Problem parameters

numModels = 5;
numParams = 3 * numModels;

# Solver Parameters


#%% Load / Generate Data

# Data link: https://drive.google.com/file/d/1M4YGsBUhgXqq_RQnniA76cY1o65NL59i
mA = readdlm("A.csv", ',', Float64);
vT = vec(readdlm("T.csv", Float64));
vB = vec(readdlm("y.csv", Float64)); #<! Manually remove ',' from the file
vX = zeros(eltype(mA), size(mA, 2));
vP = zeros(eltype(mA), numParams); #<! Parameters to optimize


## Analysis
# The model: 
# ObjFun(p) = || A x(p) - b ||₂²
# ResFun(p) = A x(p) - b
# ModelFun(p) = A x(p)

hModelFun(vT, vP) = ModelFun(mA, vT, vX, vP);
hResFun(vP) = ResFun(mA, vT, vB, vX, vP);
hObjFun(vP) = ObjFun(mA, vT, vB, vX, vP); #<! MSE

# Initialization
# Noisy initialization to make each cluster different at beginning
vP[01:05] = rand(oRng, numModels) .+ 1e-5;
vP[06:10] = 2 * rand(oRng, numModels) .- 1.0;
vP[11:15] = 2 * rand(oRng, numModels) .+ 1e-5;

# Boundaries
vL = zeros(numParams);
vL[6:10] .= minimum(vT);
vL[11:15] .= 1e-4;
vU = 1e10 * ones(numParams);
vU[6:10] .= maximum(vT);

# Each solver requires a different function:
# 1. LeastSquaresOptim -> Residual.
# 2. LsqFit -> Model.

# Solution by LeastSquaresOptim.jl (LM)
oOpt = optimize(hResFun, vP, LevenbergMarquardt(); lower = vL, upper = vU);
vPLsOptLm = oOpt.minimizer;
resAnalysis = @sprintf("MSE of LeastSquaresOptim with LM: %0.3f", hObjFun(vPLsOptLm));
println(resAnalysis);

# Solution by LeastSquaresOptim.jl (DogLeg)
oOpt = optimize(hResFun, vP, Dogleg(); lower = vL, upper = vU);
vPLsOptDog = oOpt.minimizer;
resAnalysis = @sprintf("MSE of LeastSquaresOptim with Dogleg: %0.3f", hObjFun(vPLsOptDog));
println(resAnalysis);

# Solution by LsqFit.jl (LM)
oFit = curve_fit(hModelFun, vT, vB, vP, lower = vL, upper = vU);
vPLsFLm = oFit.param;
resAnalysis = @sprintf("MSE of LsqFit with LM: %0.3f", hObjFun(vPLsFLm));
println(resAnalysis);


## Display Results

figureIdx += 1;

# Using `height = nothing, width = nothing` means current size
oConf = PlotConfig(toImageButtonOptions = attr(format = "png", height = nothing, width = nothing).fields); #<! Won't work on VS Code

oTrace1 = scatter(x = vT, y = vB, mode = "lines", text = "Measurements", name = "Measurements",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = vT, y = hModelFun(vT, vPLsOptLm), 
                  mode = "lines", text = "LeastSquaresOptim (LM)", name = "LeastSquaresOptim (LM)",
                  line = attr(width = 1.5, dash = "dot"));
oTrace3 = scatter(x = vT, y = hModelFun(vT, vPLsOptDog), 
                  mode = "lines", text = "LeastSquaresOptim (Dogleg)", name = "LeastSquaresOptim (Dogleg)",
                  line = attr(width = 1.5, dash = "dot"));
oTrace4 = scatter(x = vT, y = hModelFun(vT, vPLsFLm), 
                  mode = "lines", text = "LsqFit (LM)", name = "LsqFit (LM)",
                  line = attr(width = 1.5, dash = "dot"));
oLayout = Layout(title = "Non Linear Least Squares Fit", width = 900, height = 600, hovermode = "closest",
                 xaxis_title = "Index [t]", yaxis_title = "Value");
hP = plot([oTrace1, oTrace2, oTrace3, oTrace4], oLayout, config = oConf);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# Run Time Analysis

runTime = @belapsed optimize(hResFun, vP, LevenbergMarquardt(); lower = vL, upper = vU)  seconds = 2;
resAnalysis = @sprintf("The LeastSquaresOptim with LM solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed optimize(hResFun, vP, Dogleg(); lower = vL, upper = vU) seconds = 2;
resAnalysis = @sprintf("The LeastSquaresOptim with Dogleg solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
runTime = @belapsed curve_fit(hModelFun, vT, vB, vP, lower = vL, upper = vU) seconds = 2;
resAnalysis = @sprintf("The LsqFit with LM solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);
