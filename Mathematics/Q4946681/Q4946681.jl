# StackExchange Mathematics Q4946681
# https://math.stackexchange.com/questions/4946681
# Orthogonal Projection onto the Intersection of a Box and a Half Space.
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
# - 1.0.000     05/10/2025  Royi Avital
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
include(joinpath(juliaCodePath, "JuliaLinearAlgebra.jl"));
include(joinpath(juliaCodePath, "JuliaOptimization.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions

function ObjFun( vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    valObj = T(0.5) * sum(abs2, vX - vY);

    return valObj;
    
end

function CVXSolver( vY :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}, vA :: Vector{T}, valB :: T ) where {T <: AbstractFloat}

    numElements = length(vY);
    vX = Convex.Variable(numElements);
    
    sConvProb = minimize( T(0.5) * Convex.sumsquares(vX - vY), [vX >= vL, vX <= vU, Convex.dot(vA, vX) == valB] );
    Convex.solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vX.value);

end

function CalcBound( vY :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}, vA :: Vector{T} ) where {T <: AbstractFloat}
    # Does not work!
    # Calculates the boundaries of μ to solve (xᵢ - μ aᵢ)_[lᵢ, uᵢ]
    # No check for validity

    μmin = zero(T);
    μmax = zero(T);

    for ii in eachindex(vY)
        a = vA[ii]
        y = vY[ii]
        l = vL[ii]
        u = vU[ii]

        if a > 0
            # μ >= (y - l)/a  and  μ <= (y - u)/a
            μmax = max(μmax, (y - l) / a); #<! Must be large enough to push yᵢ - μ aᵢ ≤ lᵢ
        else
            # inequalities reverse when a < 0
            μmin = min(μmin, (y - u) / a); #<! Must be small enough to push yᵢ - μ aᵢ ≥ uᵢ
        end
    end

    return μmin, μmax;
    
end

function CheckBoxPlaneIntersection( vL :: Vector{T}, vU :: Vector{T}, vA :: Vector{T}, valB :: T ) where {T <: AbstractFloat}
    # See [Check for Intersection of Hyper Plane and a Hyperrectangle (Box)](https://stackoverflow.com/questions/70090224).

    numElements = length(vL);
    
    valMin = zero(T);
    valMax = zero(T);

    for ii in 1:numElements
        if vA[ii] >= zero(T)
            valMin += vA[ii] * vL[ii];
            valMax += vA[ii] * vU[ii];
        else
            valMin += vA[ii] * vU[ii];
            valMax += vA[ii] * vL[ii];
        end
    end

    return (valMin <= valB) && (valB <= valMax);

end

function ProjBoxHalfSpace( vY :: Vector{T}, vL :: Vector{T}, vU :: Vector{T}, vA :: Vector{T}, valB :: T ) where {T <: AbstractFloat}

    # Objective Function
    hH( μ :: T ) = dot(vA, clamp.(vY - μ * vA, vL, vU)) - valB;
    
    # Boundaries for the zero search (μmin, μmax)
    # The function is monotonically non increasing
    μmin = -T(1.0); #<! Should yield positive value
    while hH(μmin) <= T(1e-6)
        μmin *= T(2);
    end

    μmax = T(1.0); #<! Should yield negative value
    while hH(μmax) >= -T(1e-6)
        μmax *= T(2);
    end

    # Find the optimal value of μ
    μ = FindZeroBinarySearch(hH, μmin, μmax);

    return clamp.(vY - μ * vA, vL, vU);
    
end


## Parameters

# Data
numElements = 4;

## Load / Generate Data

vY   = randn(oRng, numElements);
vL   = rand(oRng, numElements) .- 1.0;
vU   = rand(oRng, numElements) .+ 1.0;
vA   = randn(oRng, numElements);
valB = randn(oRng);

while !CheckBoxPlaneIntersection(vL, vU, vA, valB)
    global vY, vL, vU, vA, valB;
    vY   = randn(oRng, numElements);
    vL   = rand(oRng, numElements) .- 1.0;
    vU   = rand(oRng, numElements) .+ 1.0;
    vA   = randn(oRng, numElements);
    valB = randn(oRng);
end

hObjFun(vX :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vX, vY);

## Analysis
# Objective: \arg \min_x 0.5 * || x - y ||_2^2 subject to l ≤ x ≤ u, aᵀx = b

# DCP Solver
methodName = "Convex.jl"

vXRef = CVXSolver(vY, vL, vU, vA, valB);
optVal = hObjFun(vXRef);

# Analytic Solution
methodName = "KKT"

vX = ProjBoxHalfSpace(vY, vL, vU, vA, valB);

# Validate Results
norm(vX - vXRef, Inf)


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