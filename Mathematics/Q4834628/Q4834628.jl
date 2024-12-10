# StackExchange Mathematics Q4834628
# https://math.stackexchange.com/questions/4834628
# Bregman Projection on Constrained Simplex.
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
# - 1.0.000     03/11/2024  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function f( vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    f  = -sum(vX);
    f += sum(vY);
    f += sum(vX .* log.(vX ./ vY));

    return f;
    
end

function ∇f( vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    vG = log.(vX) - log.(vY);

    return vG;

end

function ProjSimplexBallBox!( vX :: AbstractVector{T}, vY :: AbstractVector{T}; ballRadius :: T = T(1.0), ε :: T = T(1e-7), α :: T = T(0.0) ) where {T <: AbstractFloat}
    #TODO: Make zero allocations
    
    numElements = length(vY);

    if (length(vX) != numElements)
        throw(DimensionMismatch(lazy"The length of `vX` `vY` must match"));
    end
    
    copy!(vX, vY);

    if ((abs(sum(vY) - ballRadius) < ε) && all(vY .>= α) && all(vY .< β))
        # The input is already within the Simplex.        
        return vX;
    end

    sort!(vX); #<! TODO: Make inplace

    # Breakpoints of the piecewise function happens at xᵢ - μ = α → Search for points xᵢ - α 
    vμ         = vcat(vX[1] - ballRadius, vX .- α, vX[numElements] + ballRadius);
    hObjFun(μ) = sum(max.(vY .- μ, α)) - ballRadius;

    vObjVal = zeros(length(vμ));
    for ii = 1:length(vμ)
        vObjVal[ii] = hObjFun(vμ[ii]);
    end

    if (any(vObjVal .== zero(T)))
        μ = vμ(vObjVal .== zero(T));
    else
        # Working on when an Affine Function have the value zero
        valX1Idx = findlast(>(zero(T)), vObjVal);
        valX2Idx = findfirst(<(zero(T)), vObjVal);
    
        valX1 = vμ[valX1Idx];
        valX2 = vμ[valX2Idx];
        valY1 = vObjVal[valX1Idx];
        valY2 = vObjVal[valX2Idx];
    
        # Linear Function, Intersection with Zero
        paramA = (valY2 - valY1) / (valX2 - valX1);
        paramB = valY1 - (paramA * valX1);
        μ      = -paramB / paramA;
    end

    @. vX = max(vY - μ, α);

    return vX;

end

function ProjSimplexBallBox( vY :: AbstractVector{T}; ballRadius :: T = T(1.0), ε :: T = T(1e-7), α :: T = T(0.0) ) where {T <: AbstractFloat}
    
    numElements = length(vY);
    vX = zeros(T, numElements);

    return ProjSimplexBallBox!(vX, vY; ballRadius = ballRadius, ε = ε, α = α);

end


## Parameters

numElements = 2; #<! Number of elements
α           = 0.25;
ϵ           = 1e-5;

# Solver
numIter = 1_000;
η       = 1e-3;

# Visualization
numGridPts = 500;


## Load / Generate Data

vY = rand(oRng, numElements);


## Analysis

hF( vX :: Vector{T} ) where {T <: AbstractFloat}        = f(vX, vY);
h∇f( vX :: Vector{T} ) where {T <: AbstractFloat}       = ∇f(vX, vY);
hProjFun( vY :: Vector{T} ) where {T <: AbstractFloat}  = ProjSimplexBallBox(vY; α = α);

# Validate Gradient
vX = rand(oRng, numElements);
@assert (maximum(abs.(h∇f(vX) - CalcFunGrad(vX, hF))) <= ϵ) "The gradient calculation is not verified";

# Projected Gradient Descent

# Point in the set
vX = ones(numElements) / numElements;
mX = zeros(numElements, numIter);
mX[:, 1] .= vX;

for ii ∈ 2:numIter
    global vX;
    vX .-= η .* h∇f(mX[:, ii - 1]);
    vX   = hProjFun(vX);

    mX[:, ii] .= vX;
end


## Display Analysis

if (numElements == 2)
figureIdx += 1;
vG = LinRange(ϵ, 1.1, numGridPts);
mO = zeros(numGridPts, numGridPts);
vT = zeros(2);

for jj ∈ 1:numGridPts, ii ∈ 1:numGridPts
    # x, y notation (Matches `heatmap()`)
    vT[1] = vG[jj];
    vT[2] = vG[ii];
    mO[ii, jj] = hF(vT);
end

oShp1 = line(0.0, 1.0, 1.0, 0.0; xref = "x", yref = "y", line_color = "k");
oShp2 = rect(x0 = α, y0 = α, x1 = 1.0, y1 = 1.0; xref = "x", yref = "y", line_color = "LightSeaGreen");
oTr1 = heatmap(; x = vG, y = vG, z = log1p.(mO));
oTr2 = scatter(; x = mX[1, :], y = mX[2, :], mode = "markers", name = "Optimization Path");
oTr3 = scatter(; x = [vY[1]], y = [vY[2]], mode = "markers", marker_size = 15, name = "y");
oLayout = Layout(title = "Objective Function - Log Scale, α = $(α)", width = 600, height = 600, 
                 xaxis_title = 'x', yaxis_title = 'y',
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(x = 0.01, y = 0.99),
                 shapes = [oShp1, oShp2]);
hP = Plot([oTr1, oTr2, oTr3], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

end

