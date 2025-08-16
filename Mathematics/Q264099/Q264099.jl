# StackExchange Mathematics Q264099
# https://math.stackexchange.com/questions/264099
# Solving the Primal Kernel SVM Problem.
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
# - 1.0.000     12/08/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using ECOS;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function KernelSVM( vα :: Vector{T}, paramB :: T, mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}

    numSamples = length(vY);
    
    # Regularization
    objVal = 0.5 * λ * dot(vα, mK, vα);

    # Objective
    vH = [max(zero(T), T(1) - vY[ii] * (dot(vα, mK[:, ii]) + paramB)) for ii in 1:numSamples];
    if squareHinge
        vH = [vH[ii] * vH[ii] for ii in 1:numSamples];
    end

    objVal += sum(vH);

    # Vectorized form
    # objVal = 0.5 * λ * dot(vα, mK, vα) + sum(max.(zero(T), T(1) .- vY .* (mK * vα .+ paramB)));

    return objVal;

end

function SolveCVX( mK :: Matrix{T}, vY :: Vector{T}, λ :: T; squareHinge :: Bool = false ) where {T <: AbstractFloat}
    # Olivier Chapelle - Training a Support Vector Machine in the Primal

    # `mK` the kernel matrix
    numSamples = size(mK, 1);

    vα     = Convex.Variable(numSamples);
    paramB = Convex.Variable(1);

    vH = [Convex.pos(T(1) - vY[ii] * (Convex.dot(vα, mK[:, ii]) + paramB)) for ii in 1:numSamples];
    if squareHinge
        vH = [Convex.square(vH[ii]) for ii in 1:numSamples];
    end
    hingeLoss = Convex.sum(vcat(vH...));

    sConvProb = minimize( 0.5 * λ * Convex.quadform(vα, mK; assume_psd = true) + hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vα.value), paramB.value;
    
end

function SolveLinearSVM( mX :: Matrix{T}, vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # SciKit Learn formulation

    dataDim    = size(mX, 2);
    numSamples = size(mX, 1);

    vW     = Convex.Variable(dataDim);
    paramB = Convex.Variable(1);

    vH = [Convex.pos(T(1) - vY[ii] * (Convex.dot(vW, mX[ii, :]) + paramB)) for ii in 1:numSamples];
    if squareHinge
        vH = [Convex.square(vH[ii]) for ii in 1:numSamples];
    end
    hingeLoss = Convex.sum(vcat(vH...));

    sConvProb = minimize( 0.5 * Convex.sumsquares(vW) + λ * hingeLoss ); #<! Problem
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);

    return vec(vW.value), paramB.value;
    
end

function ProxHingeLoss( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # \arg \min_x 0.5 * || x - y ||_2^2 + λ * sum_i max(0, 1 - x_i)

    vX = zero(vY);

    for ii = 1:length(vY)
        if vY[ii] < (one(T) - λ)
            vX[ii] = vY[ii] + λ;
        elseif vY[ii] > one(T)
            vX[ii] = vY[ii];
        else
            vX[ii] = one(T);
        end
    end

    return vX;
    
end


function CalcXRangeLine( vW :: Vector{T}, tuRangeX :: Tuple{T, T}; tuRangeY :: Tuple{T, T} = tuRangeX ) where {T <: AbstractFloat}
    # Unpack input parameters for clarity
    coeffA, coeffB, coeffC = vW;
    
    minX, maxX = tuRangeX;
    minY, maxY = tuRangeY;

    # --- Handle Edge Cases ---

    # Case: Vertical Line (b ≈ 0)
    # The line equation simplifies to ax + c = 0, so x = -c/a.
    if isapprox(coeffB, zero(T))
        valX = -coeffC / coeffA;
        # Check if this constant x-value is within the box's x-bounds.
        if minX <= valX <= maxX
            # The range is a single point.
            return (valX, valX);
        else
            return nothing;
        end
    end

    # Case: Horizontal Line (a ≈ 0)
    # The line equation simplifies to by + c = 0, so y = -c/b.
    if isapprox(coeffA, zero(T))
        valY = -coeffC / coeffB;
        # Check if this constant y-value is within the box's y-bounds.
        if minY <= valY <= maxY
            # The line spans the entire width of the box.
            return (minX, maxX);
        else
            return nothing;
        end
    end

    # --- Handle General Case: Sloped Line ---

    # Calculate the x coordinates where the line intersects the horizontal boundaries
    # of the box (y = ymin and y = ymax).
    # From ax + by + c = 0  =>  x = (-by - c) / a
    valXMinY = (-coeffB * minY - coeffC) / coeffA;
    valXMaxY = (-coeffB * maxY - coeffC) / coeffA;

    # The interval for x derived from the y-constraints is [min, max] of these two points.
    if valXMinY > valXMaxY
        valXMinY, valXMaxY = valXMaxY, valXMinY;
    end

    # The final range for x is the intersection of the box's own x-range `[xmin, xmax]`
    # and the range we just derived from the y-constraints `[x_from_y_min, x_from_y_max]`.
    startX = max(minX, valXMinY);
    endX   = min(maxX, valXMaxY);

    # If the start of the final interval is greater than the end, it means the
    # intervals do not overlap, so the line does not pass through the box.
    if startX > endX
        return nothing;
    else
        return (startX, endX);
    end

end

function CalcLineVal( vW :: Vector{T}, vX :: Vector{T}) where {T <: AbstractFloat}
    # Assumes `vW[2] ≠ 0`

    vY = (-vW[1] .* vX .- vW[3]) ./ vW[2];

    return vY;
    
end


## Parameters

# Data

# Solver
λ           = 1.0;
squareHinge = false;
numIter     = 150_000;
η           = 2.5e-2; #<! Step Size

# Visualization
tuRange = (-3.0, 3.0);
numGridPts = 1000;


## Load / Generate Data

# From SK Learn Example (https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)
mX = [
     0.4 -0.7;
    -1.5 -1.0;
    -1.4 -0.9;
    -1.3 -1.2;
    -1.1 -0.2;
    -1.2 -0.4;
    -0.5  1.2;
    -1.5  2.1;
     1.0  1.0;
     1.3  0.8;
     1.2  0.5;
     0.2 -2.0;
     0.5 -2.4;
     0.2 -2.3;
     0.0 -2.7;
     1.3  2.1;
];

vY = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

numSamples = length(vY);


## Analysis

mK = mX * mX';
mK = 0.5 * (mK + mK');
vα, paramBB = SolveCVX(mK, vY, λ; squareHinge = squareHinge);

# vSuppVecIdx = findall(vY .* (mK' * vα .+ paramBB) .≈ 1.0); #<! Too restrictive
vSuppVecIdx = findall(isapprox.(vY .* (mK' * vα .+ paramBB), 1.0; atol = 5e-3));

# The solutions obey:
# mK * vα + b == mX * vw + b0
# b == b0
# Hence: mK * vα = mX * vw
#        mX * mX' vα = mX * vW => mX' * vα = vW
# vW = mX' * vα;
# vW = [vW[1], vW[2], paramBB];

# Sub Gradient
vβ     = randn(oRng, numSamples);
paramB = 0.1;

hGradβ( vβ :: Vector{T}, paramB :: T ) where {T <: AbstractFloat} = -((vY .* ((mK * vβ) .+ paramB)) .< T(1.0)) .* (mK' * vY) + λ * mK * vβ;
hGradB( vβ :: Vector{T}, paramB :: T ) where {T <: AbstractFloat} = -((vY .* ((mK * vβ) .+ paramB)) .< T(1.0))' * vY;

for ii in 1:numIter
    global vβ, paramB;
    # global paramB;

    ηₖ  = η / ii;
    vβ0 = copy(vβ);

    vβ    .-= ηₖ * hGradβ(vβ0, paramB);
    paramB -= ηₖ * hGradB(vβ0, paramB);
end

vW = mX' * vβ;
vW = [vW[1], vW[2], paramB];

println(mX' * (vβ - vα));
println(KernelSVM(vα, paramBB, mK, vY, λ));
println(KernelSVM(vβ, paramB, mK, vY, λ));

# ADMM
mE = diagm(numSamples, numSamples + 1, ones(numSamples));
mK1 = [mK ones(numSamples)];
mK0 = mE' * mK * mE;
mA  = diagm(vY) * mK1;
mAA = mA' * mA;
# vAy = mA' * vY;
hProxF( vY :: Vector{T}, ρ :: T ) where {T <: AbstractFloat} = (λ * mK0 + ρ * mAA) \ (ρ * mA' * vY);
hProxG( vY :: Vector{T}, λ :: T ) where {T <: AbstractFloat} = ProxHingeLoss(vY, λ);

vγ = zeros(numSamples + 1);

vγ = ADMM(vγ, mA, hProxF, hProxG, 1_000);
vW = [mX' * vγ[1:numSamples]; vγ[end]];
println(KernelSVM(vγ[1:numSamples], vγ[end], mK, vY, λ));


## Display Analysis

figureIdx += 1;

startX, endX = CalcXRangeLine(vW, tuRange);
vXX = collect(LinRange(startX, endX, numGridPts));
vYY = CalcLineVal(vW, vXX);

sTr1 = scatter(x = mX[:, 1], y = mX[:, 2], mode = "markers", text = "Data Samples", name = "Data Samples",
                marker = attr(size = 10, color = vY));
sTr2 = scatter(x = vXX, y = vYY, 
                mode = "lines", text = "Classifier", name = "Classifier",
                line = attr(width = 2.5));
sTr3 = scatter(x = mX[vSuppVecIdx, 1], y = mX[vSuppVecIdx, 2], mode = "markers", text = "Support Vectors", name = "Support Vectors",
                marker = attr(size = 20, color = "rgba(0, 0, 0, 0)", line = attr(color = "k", width = 2)));
sLayout = Layout(title = "The Data Samples and Classifier", width = 600, height = 600, hovermode = "closest",
                xaxis_title = "t", yaxis_title = "y", xaxis_range = tuRange, yaxis_range = tuRange,
                legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

