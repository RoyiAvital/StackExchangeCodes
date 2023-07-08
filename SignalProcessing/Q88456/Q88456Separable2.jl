# StackExchange Signal Processing Q88456
# https://dsp.stackexchange.com/questions/88456
# Solving Inverse Problem of Multiple Pulses
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
# - 1.0.000     24/06/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using DelimitedFiles;
using FileIO;
import FreeType;
using LinearAlgebra;
using MKL;
using UnicodePlots;

## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = true;

## Functions

hObjFun( mX, mR, mQ ) = (0.5 .* sum(abs2, ((mR * mX) .- mQ)));
∇ObjFun( mX, mRR, mRQ, λ ) = (mRR * mX) .- mRQ;

function hObjFun!( mX, mR, mQ, mRX )
    LinearAlgebra.mul!(mRX, mR, mX);
    # Triangular multiplication (Slower)
    # mRX .= mX;
    # LinearAlgebra.BLAS.trmm!('L', 'L', 'N', 'N', 1.0, mR, mRX);  

    mRX .-= mQ;
    return 0.5 .* (sum(abs2, mRX));

end

function ∇ObjFun!( ∇mX, mX, mRR, mRQ, mRRX )

    LinearAlgebra.BLAS.symm!('L', 'L', 1.0, mRR, mX, 0.0, mRRX);

    ∇mX .= mRRX .- mRQ;

    return ∇mX;

end


## Parameters

# Data
mAFileName = "mA.csv";
mRFileName = "mR.csv";
mYFileName = "mY.csv";

# Model
λ = 5;
η = 0.0000003
minValThr = 0;

# Gradient Descent
numIter = 500;


## Load / Generate Data

# mA = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/A.csv"), ',', Float64);
# mR = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/R.csv"), ',', Float64);
# mY = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/signals.csv"), ',', Float64);
mA = readdlm(mAFileName, ',', Float64);
mR = readdlm(mRFileName, ',', Float64);
mY = readdlm(mYFileName, ',', Float64);


## Analysis
# Step I
# Solve \arg \min_Q || Q A - Y ||_F^2

mQ = mY / mA;

# Step II
# Solve \arg \min_X || R X - Q ||_F^2 
#       subject to X_ij >= thr

# Solving using accelerated projected gradient descent.

# Pre Calculation
mRR = mR' * mR;
mRQ = mR' * mQ;
vObjVal = zeros(numIter);

# Buffers
mX      = zeros(size(mR, 2), size(mA, 1));
mXPrev  = zeros(size(mR, 2), size(mA, 1));
mZ      = zeros(size(mR, 2), size(mA, 1));
∇mZ     = zeros(size(mR, 2), size(mA, 1));

mRX  = zeros(size(mR, 1), size(mX, 2));
mRRX = zeros(size(mRR, 1), size(mX, 2));

runTime = @elapsed for ii ∈ 1:numIter
    # FISTA (Nesterov) Accelerated
    # local ∇mZ = ∇ObjFun(mZ, mRR, mAA, mRYA, λ);

    ∇ObjFun!(∇mZ, mZ, mRR, mRQ, mRRX);

    mXPrev .= mX;
    mX .= mZ .- (η .* ∇mZ);
    mX .= max.(mZ .- (η .* ∇mZ), minValThr);
    # vObjVal[ii] = hObjFun(mX, mR, mA, mY, λ);
    # vObjVal[ii] = hObjFun!(mX, mR, mQ, mRX);

    fistaStepSize = (ii - 1) / (ii + 2);

    mZ .= mX .+ (fistaStepSize .* (mX .- mXPrev))
end


# display(lineplot(1:numIter, log.(vObjVal)));
println("Objective Function final value: $(hObjFun!(mX, mR, mQ, mRX))");
println("Total runtime: $(runTime) [Sec]");
writedlm("mX.csv",  mX, ',');


