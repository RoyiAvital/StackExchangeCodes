# StackExchange Mathematics Q4025123
# https://math.stackexchange.com/questions/4025123
# Path Optimization (Minimize the Sum of Euclidean Distance) Using the Sub Gradient Method.
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
# - 1.0.000     19/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
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

function SolvePathConvex( vS :: Vector{T}, vE :: Vector{T}, numPts :: N ) where {T <: AbstractFloat, N <: Integer}

    mX = Variable(2, numPts);
    vV = [Convex.norm2(mX[:, ii] - mX[:, ii - 1]) for ii ∈ 2:numPts];
    
    sConvProb = minimize( Convex.sum(vcat(vV...)), [mX[:, 1] == vS, mX[:, numPts] == vE] ); #<! https://github.com/jump-dev/Convex.jl/issues/722
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return mX.value;

end

function ObjFun( mX :: Matrix{T} ) where {T <: AbstractFloat}

    numPts = size(mX, 2);
    
    valObj = zero(T);
    for ii ∈ 2:numPts
        valObj += norm(mX[:, ii] - mX[:, ii - 1]);
    end

    return valObj;

end

function ∇ObjFun( mX :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim = size(mX, 1);
    numPts  = size(mX, 2);
    
    mG = zeros(T, dataDim, numPts);

    if norm(mX[:, 2] - mX[:, 1]) > zero(T)
        # Contribution of `mX[:, 1]` to `mG[:, 1]`
        mG[:, 1] -= (mX[:, 2] - mX[:, 1]) ./ norm(mX[:, 2] - mX[:, 1]);
    end
    
    for ii ∈ 2:(numPts - 1)
        if norm(mX[:, ii] - mX[:, ii - 1]) > zero(T)
            # @infiltrate
            # Contribution of `mX[:, ii]` to `mG[:, ii]` in `mX[:, ii] - mX[:, ii - 1]`
            mG[:, ii] += (mX[:, ii] - mX[:, ii - 1]) ./ norm(mX[:, ii] - mX[:, ii - 1]);
        end
        if norm(mX[:, ii + 1] - mX[:, ii]) > zero(T)
            # @infiltrate
            # Contribution of `mX[:, ii]` to `mG[:, ii]` in `mX[:, ii + 1] - mX[:, ii]`
            mG[:, ii] -= (mX[:, ii + 1] - mX[:, ii]) ./ norm(mX[:, ii + 1] - mX[:, ii]);
        end
    end

    ii = numPts;
    if norm(mX[:, ii] - mX[:, ii - 1]) > zero(T)
        # Contribution of `mX[:, ii]` to `mG[:, ii]`
        mG[:, ii] += (mX[:, ii] - mX[:, ii - 1]) ./ norm(mX[:, ii] - mX[:, ii - 1]);
    end

    return mG;

end

function ProjFun( mX :: Matrix{T}, vS :: Vector{T}, vE :: Vector{T} ) where {T <: AbstractFloat}

    numPts  = size(mX, 2);

    mX[:, 1]      .= vS;
    mX[:, numPts] .= vE;

    return mX;

end

# To evaluate the Gradient numerically
hF( vX :: Vector{T} ) where {T <: AbstractFloat} = ObjFun(reshape(vX, 2, numPts));

## Parameters

# Path
numPts   = 10;
vStartPt = [0.0, 0.0];
vEndPt   = [3.0, 2.0];

# Gradient Verification
ϵ = 1e-5;

# Solver
numIter = 250_000;
η       = 2e-5;

## Load / Generate Data

mX0            = rand(2, numPts) .* vEndPt;
mX0[:, 1]      = vStartPt;
mX0[:, numPts] = vEndPt;


## Analysis

# Reference solution
mXRef = SolvePathConvex(vStartPt, vEndPt, numPts);

# Verify the Gradient
mT    = randn(2, numPts);
mGRef = reshape(CalcFunGrad(mT[:], hF), 2, numPts);
mG    = ∇ObjFun(mT);

println("Gradient verified: $(norm(mG - mGRef, Inf) < ϵ)");

# Projected Sub Gradient
hProj( mX :: Matrix{T} ) where {T <: AbstractFloat} = ProjFun(mX, vStartPt, vEndPt);

mX = copy(mX0);
GradientDescentAccelerated(mX, numIter, η, ∇ObjFun; ProjFun = hProj);


## Display Analysis

figureIdx += 1;

sTr1 = scatter(; x = mX[1, :], y = mX[2, :], mode = "markers+lines", 
              line = attr(width = 2.75, color = "#636EFAFF"),
              marker = attr(size = 8),
              name = "Sub Gradient", text = "Sub Gradient");
sTr2 = scatter(; x = mXRef[1, :], y = mXRef[2, :], mode = "markers+lines", 
              line = attr(width = 2.25, color = "#EF553B80"),
              marker = attr(size = 6),
              name = "Reference", text = "Reference");
sTr3 = scatter(; x = mX0[1, :], y = mX0[2, :], mode = "markers+lines", 
              line = attr(width = 2, color = "#00CC9620"),
              name = "Initialization", text = "Initialization");
sLayout = Layout(title = "Optimal Path (L2 Norm)", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01));
hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end


