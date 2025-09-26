# StackExchange Mathematics Q2246150
# https://math.stackexchange.com/questions/2246150
# Solving Kernel SVM with a Large Kernel Matrix.
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
# - 1.0.000     23/09/2025  Royi Avital
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
using FastLapackInterface; #<! Required for Optimization
using Krylov;
using LinearOperators;
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
# include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( vβ :: Vector{T}, valB :: T, mK :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # (λ / 2) * βᵀ K β + 0.5 * sum_i max^2( 0, 1 - y_i * (k_iᵀ β + b) )

    regLoss = T(0.5) * λ * dot(vβ, mK, vβ);
    hingeLoss = zero(T);

    for ii in 1:numSamples
        valS       = vY[ii] * (dot(vK, vβ) + valB);
        valM       = max(zero(T), one(T) - valS);
        hingeLoss += valM * valM; #<! Squared Hinge Loss
    end

    return regLoss + T(0.5) * hingeLoss;
    
end

function SolveCVX( mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}
    # Olivier Chapelle - Training a Support Vector Machine in the Primal

    # `mK` the kernel matrix
    numSamples = size(mK, 1);

    vα     = Convex.Variable(numSamples);
    paramB = Convex.Variable(1);

    # Loop Form
    # vH = [Convex.pos(T(1) - vY[ii] * (Convex.dot(vα, mK[:, ii]) + paramB)) for ii in 1:numSamples];
    # if squareHinge
    #     vH = [Convex.square(vH[ii]) for ii in 1:numSamples];
    # end
    # hingeLoss = Convex.sum(vcat(vH...));

    # Vectorized Form
    if squareHinge
        hingeLoss = T(0.5) * Convex.sum(Convex.square(Convex.pos(T(1) - vY .* (mK * vα + paramB))));
    else
        hingeLoss = Convex.sum(Convex.pos(T(1) - vY .* (mK * vα + paramB)));
    end

    sConvProb = minimize( T(0.5) * λ * Convex.quadform(vα, mK; assume_psd = true) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vα.value), paramB.value;
    
end

function GetKi!( vK :: Vector{T}, ii :: N, mX :: Matrix{T}, vB :: Vector{T}, σ :: T ) where {T <: AbstractFloat, N <: Integer}

    numSamples = size(mX, 2); #<! `mK` is (numSamples, numSamples)
    denFactor  = T(2) * σ * σ;

    for jj in 1:numSamples
        vB .= mX[:, ii] .- mX[:, jj];
        vK[jj] = exp(-sum(abs2, vB) / denFactor);
    end

    return vK;

end

function MulK!( mX :: Matrix{T}, vK :: Vector{T}, vB :: Vector{T}, vI :: AbstractVector{T}, vO :: AbstractVector{T}, σ :: T; α :: T = one(T), β :: T = zero(T) ) where {T <: AbstractFloat}
    # Calculates mC = α * mK * mB + β * mC
    # Calculates vO = α * mK * vI + β * vO

    # `vK` - Buffer for a row / column of the kernel matrix (numSamples, 1).
    # `vB` - Buffer (numSamples, 1).

    if β == zero(T) #<! Use `vO` only as output buffer
        # Better 2 variations of the loop, yet for simple way to deal with uninitialized arrays.
        fill!(vO, zero(T));
    end

    numSamples = size(mX, 2); #<! `mK` is (numSamples, numSamples)

    for ii in 1:numSamples
        vK = GetKi!(vK, ii, mX, vB, σ);
        vO[ii] = α * dot(vK, vI) + β * vO[ii];
    end

end

function MulKβ!( mX :: Matrix{T}, vS :: Vector{N}, vK :: Vector{T}, vB :: Vector{T}, vβ :: Vector{T}, vKβ :: Vector{T}, σ :: T ) where {T <: AbstractFloat, N <: Integer}
    # Calculates vKβ = mK * vβ
    # For all rows of mK yet subset of the columns (`vS`)

    # `vK` - Buffer for a row / column of the kernel matrix (numSamples, 1).
    # `vB` - Buffer (numSamples, 1).

    numSamples = size(mX, 2); #<! `mK` is (numSamples, numSamples)
    vKK        = view(vK, vS);
    vββ        = view(vβ, 1:(length(vβ) - 1)); #<! Without the last element (Bias)

    for ii in 1:numSamples
        vK      = GetKi!(vK, ii, mX, vB, σ);
        vKβ[ii] = dot(vKK, vββ);
    end

end

function CalcK( mX :: Matrix{T}, σ :: T ) where {T <: AbstractFloat}

    dataDim = size(mX, 1);
    numSamples = size(mX, 2); #<! `mK` is (numSamples, numSamples)
    denFactor = T(2) * σ * σ;
    mK = ones(T, numSamples, numSamples); #<! Diagonal of 1
    vB = zeros(T, dataDim); #<! Buffer

    for jj in 2:numSamples
        for ii in 1:(jj - 1)
            vB .= mX[:, ii] .- mX[:, jj];
            mK[ii, jj] = exp(-sum(abs2, vB) / denFactor);
            mK[jj, ii] = mK[ii, jj];
        end
    end

    return mK;

end

function CalcGrad!( vG :: Vector{T}, vβ :: Vector{T}, valB :: T, mX :: Matrix{T}, vY :: Vector{N}, vK :: Vector{T}, vB :: Vector{T}, σ :: T, λ :: T ) where {T <: AbstractFloat, N <: Integer}
    # Gradient of (λ / 2) * βᵀ K β + 0.5 * sum_i max^2( 0, 1 - y_i * (k_iᵀ β + b) )
    # The squaring of the Hinge Loss makes it smooth.
    # For the vector and Bias separated

    MulK!(mX, vK, vB, vβ, vG, σ);
    vG .*= λ;
    valG = zero(T); #<! Gradient with respect to b

    for ii in 1:numSamples
        vK = GetKi!(vK, ii, mX, vB, σ); #<! Extract the i -th row / column of `mK`
        valS = vY[ii] * (dot(vK, vβ) + valB);
        if valS < one(T)
            vG .+= vK .* vY[ii] .* (valS - one(T));

            valG += vY[ii] .* (valS - one(T));
        end
    end

    return vG, valG;

end

function CalcGrad!( vG :: Vector{T}, vβ :: Vector{T}, mX :: Matrix{T}, vY :: Vector{N}, vK :: Vector{T}, vB :: Vector{T}, σ :: T, λ :: T ) where {T <: AbstractFloat, N <: Integer}
    # Gradient of (λ / 2) * βᵀ K β + 0.5 * sum_i max^2( 0, 1 - y_i * (k_iᵀ β + b) )
    # The squaring of the Hinge Loss makes it smooth.
    # Here b = vβ[end]

    numSamples = size(mX, 2);
    vGG = view(vG, 1:numSamples);
    vββ = view(vβ, 1:numSamples);


    MulK!(mX, vK, vB, vββ, vGG, σ);
    vGG .*= λ;
    vG[end] = zero(T); #<! Gradient with respect to b

    for ii in 1:numSamples
        vK = GetKi!(vK, ii, mX, vB, σ); #<! Extract the i -th row / column of `mK`
        valS = vY[ii] * (dot(vK, vββ) + valB);
        if valS < one(T)
            vGG .+= vK .* vY[ii] .* (valS - one(T));

            vG[end] += vY[ii] .* (valS - one(T));
        end
    end

    return vG;

end

function MulH!( mX :: Matrix{T}, vS :: Vector{N}, vK :: Vector{T}, vB :: Vector{T}, vI :: Vector{T}, vO :: Vector{T}, σ :: T, λ :: T; α :: T = T(1), β :: T = T(0) ) where {T <: AbstractFloat, N <: Integer}
    # Operator which: [ mK + λ I, 1; 1 0 ]
    # `vS` is `Vector{Bool}`, yet `Bool <: Integer`.

    numSamples = size(mX, 2);
    numSv = sum(vS);
    # All should obey (numSv + 1) == length(vO) == length(vI);
    vII = view(vI, 1:numSv);
    vKK = view(vK, vS);

    if β == zero(T)
        fill!(vO, zero(T));
    end

    jj = 0;
    for ii in 1:numSamples
        if vS[ii]
            # The ii -th row / column of `mK` is used
            jj    += 1;
            vK     = GetKi!(vK, ii, mX, vB, σ);
            vO[jj] = α * (dot(vKK, vII) + λ * vI[jj] + vI[end]) + β * vO[jj];
        end
    end

    jj += 1;
    vO[jj] = α * sum(vII) + β * vO[jj];

end


function SolveKernelSvm( mX :: Matrix{T}, vY :: Vector{M}, σ :: T, λ :: T; numIter :: N = 500 ) where {T <: AbstractFloat, M <: Integer, N <: Integer}
    # Solve (λ / 2) * βᵀ K β + 0.5 * sum_i max^2( 0, 1 - y_i * (k_iᵀ β + b) )
    # Squaring the Hinge Loss makes it smooth.
    # Using Newton Method as in Olivier Chapelle - Training a Support Vector Machine in the Primal.
    # This implementation does not use the matrix K explicitly.

    numSamplesThr = 100;

    dataDim    = size(mX, 1);
    numSamples = size(mX, 2);

    vB = zeros(T, dataDim);
    vK = zeros(T, numSamples);
    vI = zeros(T, numSamples + 1);
    vO = zeros(T, numSamples + 1);
    vZ = zeros(T, numSamples + 1);
    for ii = 1:numSamples
        vZ[ii] = T(vY[ii]);
    end

    vS  = zeros(Bool, numSamples); #<! Indicator of Support Vectors
    vSS = ones(Bool, numSamples + 1); #<! Indicator of Support Vectors (Includes bias)
    vβ  = zeros(T, numSamples + 1); #<! Last element as b
    vKβ = zeros(T, numSamples); #<! mK * vβ[1:numSamples]

    if (numSamples > numSamplesThr) && (1 < 0)
        # Solve sub set for initialization
        # Does not seem to reduce the number of needed iterations -> Hence deactivated
        vIdx = randperm(numSamples)[1:numSamplesThr]; #<! Not efficient, better use `StatBase` sample(1:100000, 100, replace = false);
        vβᵢ =  SolveKernelSvm(mX[:, vIdx], vY[vIdx], σ, λ; numIter = numIter);
        vβ[vIdx] .= vβᵢ[1:numSamplesThr];
        vβ[end]   = vβᵢ[end];
        vS[vIdx] .= true;
        MulKβ!(mX, vS, vK, vB, vβᵢ, vKβ, σ);        
    end

    vS1 = copy(vS); #<! Previous cycle

    for ii in 1:numIter
        copy!(vS1, vS); #<! Save previous iteration

        numSv = 0;
        for jj in 1:numSamples
            vS[jj]  = (T(vY[jj]) * (vKβ[jj] + vβ[end])) < one(T); #<! Test for active vector
            vSS[jj] = vS[jj];
            numSv  += ifelse(vS[jj], 1, 0);
        end
        if (ii > 1) && (vS == vS1) #<! For `AbstractVectors`, `all(vA .== vB)` is equivalent to `vA == vB`
            # No change in support vectors -> Convergence
            # println("Number of iterations: $(ii)");
            break;
        end

        mH = LinearOperator(T, numSv + 1, numSv + 1, true, true, (vO, vI, α, β) -> MulH!(mX, vS, vK, vB, vI, vO, σ, λ; α = α, β = β));

        (vβᵢ, _) = Krylov.minres(mH, view(vZ, vSS), vβ[vSS]); #<! Use vβ[vS] for initialization (Take care of `b` / vβ[end])
        # (vβᵢ, _) = Krylov.cg(mH, view(vZ, vSS)); #<! Use vβ[vS] for initialization (Take care of `b` / vβ[end])

        # vβ[vSS] .= vβᵢ; #<! I am not sure this non allocates -> Loop
        jj = 1; #<! Ensure valid index as `ifelse()` evaluates both its arguments
        for kk = 1:numSamples
            vβ[kk] = ifelse(vS[kk], vβᵢ[jj], zero(T));
            jj    += ifelse(vS[kk], 1, 0);
        end
        vβ[end] = vβᵢ[end]; #<! Bias term

        MulKβ!(mX, vS, vK, vB, vβᵢ, vKβ, σ);

    end

    return vβ;
    
end


## Parameters

# Data
csvFileName = raw"MNIST12x12.csv";
numSamples  = 500; #<! Up to ~13_000 (70_000 in total, ~7000 per digit)
tuDigits    = (0, 4); #<! Only 2 digits to take (Binary Classification)

# SVM Model
σ = 1.0;
λ = 0.1;

# Solvers
numIterations = 25_000;


## Load / Generate Data

mD = readdlm(csvFileName, ',', UInt8; skipstart = 1);
mX = mD[:, 1:(end - 1)]; #<! Data in Python style (Each row is a sample)
mX = collect(mX'); #<! Each sample as a column
mX = Float64.(mX) / 255.0; #<! Normalize into [0, 1]
vY = Int8.(mD[:, end]);

vIdx = (vY .== tuDigits[1]) .|| (vY .== tuDigits[2]);
vY   = vY[vIdx];
mX   = mX[:, vIdx];
replace!(vY, tuDigits[1] => -1, tuDigits[2] => 1);

mX = mX[:, 1:numSamples];
vY = vY[1:numSamples];

# DisplayImage(collect(reshape(mX[:, 1], 12, 12)'), tuImgSize = (112, 112), titleStr = "$(vY[1])")

# mX = [
#      0.4 -0.7;
#     -1.5 -1.0;
#     -1.4 -0.9;
#     -1.3 -1.2;
#     -1.1 -0.2;
#     -1.2 -0.4;
#     -0.5  1.2;
#     -1.5  2.1;
#      1.0  1.0;
#      1.3  0.8;
#      1.2  0.5;
#      0.2 -2.0;
#      0.5 -2.4;
#      0.2 -2.3;
#      0.0 -2.7;
#      1.3  2.1;
# ];
# mX = collect(mX');
# vY = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
# vY = Int8.(vY);

# numSamples = length(vY);

dataDim = size(mX, 1);

# Images are stored in Row Major form (Python) -> Use transpose after reshaping into a matrix


## Analysis

# hObjFun(vX :: Vector{T}) where {T <: AbstractFloat} = ObjFun(mA, vX, vB);

mK = CalcK(mX, σ);
vT = rand(numSamples);
vZ = rand(numSamples + 1);

vORef = mK * vT;

vK = zeros(numSamples);
vB = zeros(dataDim);
vO = zeros(numSamples);

MulK!(mX, vK, vB, vT, vO, σ);

maximum(abs.(vO - vORef))

vS = ones(Bool, numSamples);
vS = rand(Bool, numSamples);
vSS = ones(Bool, numSamples + 1);
vSS[1:numSamples] .= vS;

mKK = mK[vS, vS];
numSv = sum(vS);

mH = [mKK + λ*I ones(numSv); ones(numSv)' 0.0];
oH = LinearOperator(Float64, numSv + 1, numSv + 1, true, true, (vO, vI, α, β) -> MulH!(mX, vS, vK, vB, vI, vO, σ, λ; α = α, β = β));

maximum(abs.(mH * vZ[vSS] - oH * vZ[vSS]))

vβ, valB = SolveCVX(mK, Float64.(vY), λ; squareHinge = true);

vG = zeros(numSamples);
CalcGrad!(vG, vβ, valB, mX, vY, vK, vB, σ, λ)

vβRef = [vβ; valB];

vG = zeros(numSamples + 1);
CalcGrad!(vG, vβRef, mX, vY, vK, vB, σ, λ)

vβ = SolveKernelSvm(mX, vY, σ, λ)

maximum(abs.(vβ - vβRef))


## Display Results

# figureIdx += 1;

# vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# # shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
# for (ii, methodName) in enumerate(keys(dSolvers))
#     vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
#                mode = "lines", line = attr(width = 3.0),
#                text = methodName, name = methodName);
# end
# sLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = raw"$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]");

# hP = Plot(vTr, sLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
# end

