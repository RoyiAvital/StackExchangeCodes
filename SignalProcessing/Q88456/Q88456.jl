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

hObjFun( mX, mR, mA, mY, λ ) = (0.5 .* sum(abs2, ((mR * mX * mA) .- mY))) + ((0.5 * λ) .* sum(abs2, mX));
∇ObjFun( mX, mRR, mAA, mRYA, λ ) = ((mRR * mX * mAA) .- mRYA .+ (λ .* mX));

function hObjFun!( mX, mR, mA, mY, λ, mXA,  mRXA)
    LinearAlgebra.mul!(mXA, mX, mA);
    LinearAlgebra.mul!(mRXA, mR, mXA);
    # Traiangular multiplication (Slower)
    # mRXA .= mXA;
    # LinearAlgebra.BLAS.trmm!('L', 'L', 'N', 'N', 1.0, mR, mRXA);  

    mRXA .-= mY;
    return 0.5 .* (sum(abs2, mRXA) .+ (λ .* sum(abs2, mX)));

end

function ∇ObjFun!( ∇mX, mX, mRR, mAA, mRYA, λ, mXAA, mRRXAA )

    # LinearAlgebra.mul!(mXAA, mX, mAA);
    # LinearAlgebra.mul!(mRRXAA, mRR, mXAA);

    LinearAlgebra.BLAS.symm!('R', 'L', 1.0, mAA, mX, 0.0, mXAA);
    LinearAlgebra.BLAS.symm!('L', 'L', 1.0, mRR, mXAA, 0.0, mRRXAA);

    ∇mX .= mRRXAA .- mRYA .+ (λ .* mX);

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
numIter = 3_000;


## Load / Generate Data

# mA = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/A.csv"), ',', Float64);
# mR = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/R.csv"), ',', Float64);
# mY = readdlm(download("https://github.com/DJDuque/SP_Q/raw/main/signals.csv"), ',', Float64);
mA = readdlm(mAFileName, ',', Float64);
mR = readdlm(mRFileName, ',', Float64);
mY = readdlm(mYFileName, ',', Float64);


## Analysis

mAA  = mA * mA';
mRYA = mR' * mY * mA';
mRR  = mR' * mR;
vObjVal = zeros(numIter);

mX      = zeros(size(mR, 2), size(mA, 1));
mXPrev  = zeros(size(mR, 2), size(mA, 1));
mZ      = zeros(size(mR, 2), size(mA, 1));
∇mZ     = zeros(size(mR, 2), size(mA, 1));

mXA  = zeros(size(mX, 1), size(mA, 2));
mRXA = zeros(size(mR, 1), size(mXA, 2));

mXAA   = zeros(size(mX, 1), size(mAA, 2));
mRRXAA = zeros(size(mRR, 1), size(mXAA, 2));

# global tK::Float64 = 1; 

runTime = @elapsed for ii ∈ 1:numIter
    # FISTA (Nesterov) Accelerated
    # local ∇mZ = ∇ObjFun(mZ, mRR, mAA, mRYA, λ);

    ∇ObjFun!(∇mZ, mZ, mRR, mAA, mRYA, λ, mXAA, mRRXAA);

    mXPrev .= mX;
    mX .= mZ .- (η .* ∇mZ);
    mX .= max.(mZ .- (η .* ∇mZ), minValThr);
    # vObjVal[ii] = hObjFun(mX, mR, mA, mY, λ);
    vObjVal[ii] = hObjFun!(mX, mR, mA, mY, λ, mXA, mRXA);

    fistaStepSize = (ii - 1) / (ii + 2);

    mZ .= mX .+ (fistaStepSize .* (mX .- mXPrev))
end


display(lineplot(1:numIter, log.(vObjVal)));
println("Objective Function final value: $(vObjVal[end])");
println("Total runtime: $(runTime) [Sec]");
writedlm("mX.csv",  mX, ',');


