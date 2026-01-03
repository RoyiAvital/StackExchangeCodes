# StackExchange Mathematics Q5117689
# https://math.stackexchange.com/questions/5117689
# Minimize the Sum of Euclidean Distance to a Set of Points in 3D.
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
# - 1.0.000     02/01/2026  Royi Avital
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

function CVXSolver( mP :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim = size(mP, 1);
    numPts  = size(mP, 2);

    vP = Convex.Variable(dataDim);
    sConvProb = minimize( sum([Convex.norm(vP - mP[:, ii]) for ii in 1:numPts]) );
    solve!(sConvProb, ECOS.Optimizer; silent = true);
    
    return vec(vP.value);
    
end

function ObjFun( vP :: Vector{T}, mP :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim = size(mP, 1);
    numPts  = size(mP, 2);
    
    sumDist = zero(T);
    vD      = zeros(T, dataDim);

    for ii in 1:numPts
        vD .= vP .- view(mP, :, ii);
        sumDist += norm(vD);
    end

    return sumDist;
    
end

function ∇ObjFun( vP :: Vector{T}, mP :: Matrix{T} ) where {T <: AbstractFloat}

    dataDim = size(mP, 1);
    numPts  = size(mP, 2);

    vD = zeros(T, dataDim);
    vG = zeros(T, dataDim);

    for ii in 1:numPts
        vD .= vP .- view(mP, :, ii);
        vG .+= vD ./ norm(vD);
    end

    return vG;
    
end

function SampleUnitSphere( dataDim :: N; dataType :: Type{T}, oRng :: R = Random.default_rng() ) where {N <: Integer, T <: AbstractFloat, R <: AbstractRNG}

    vP = randn(oRng, dataType, dataDim);
    λ = norm(vP);
    vP ./= λ;

    return vP;

end

function GradientDescentAccelerated( mXi :: Matrix{T}, η :: T, ∇ObjFun :: Function ) where {T <: AbstractFloat}
    # This variation allocates memory.
    # No requirements from ∇ObjFun, ProjFun to be allocations free.

    dataDim = size(mXi, 1);
    numIter = size(mXi, 2);

    vW = zeros(T, dataDim);
    vZ = copy(mXi[:, 1]);

    ∇vZ = zeros(T, dataDim);

    for ii ∈ 2:numIter
        # FISTA (Nesterov) Accelerated
    
        ∇vZ = ∇ObjFun(vZ);
    
        vW .= @views mXi[:, ii - 1]; #<! Previous iteration
        @views mXi[:, ii] .= vZ .- (η .* ∇vZ);
    
        fistaStepSize = (ii - 1) / (ii + 2);
    
        @views vZ .= mXi[:, ii] .+ (fistaStepSize .* (mXi[:, ii] .- vW));
    end

    return mXi;

end


## Parameters

# Data
dataDim = 3;
numPts = 10;

# Solver
numIterations = 250;
η = 5e-4;


## Load / Generate Data

mP = zeros(dataDim, numPts);
for ii in 1:numPts
    @views mP[:, ii] .= SampleUnitSphere(3; dataType = eltype(mP), oRng = oRng);
end

## Analysis

hObjFun(vP :: Vector{T}) where {T <: AbstractFloat} = ObjFun(vP, mP);
h∇ObjFun(vP :: Vector{T}) where {T <: AbstractFloat} = ∇ObjFun(vP, mP);

# Verify Gradient
vP0 = rand(oRng, dataDim);
vGRef = CalcFunGrad(vP0, hObjFun);
vG = h∇ObjFun(vP0);

println(maximum(abs.(vG - vGRef)));

# Reference Solution
vPRef = CVXSolver(mP);

# Accelerated Gradient Descent Solution
mXi = zeros(dataDim, numIterations);
mXi[:, 1] = mean(mP; dims = 2);
mXi = GradientDescentAccelerated(mXi, η, h∇ObjFun);


## Display Results

figureIdx += 1;

sTr1 = scatter3d(; x = mP[1, :], y = mP[2, :], z = mP[3, :], mode = "markers", 
               marker_size = 7,
               name = "Points Set", text = "Points Set");
sTr2 = scatter3d(; x = mXi[1, :], y = mXi[2, :], z = mXi[3, :], mode = "markers", 
               marker_size = 3,
               name = "Optimization Path", text = "Optimization Path");
sTr3 = scatter3d(; x = [vPRef[1]], y = [vPRef[2]], z = [vPRef[3]], mode = "markers", 
               marker_size = 5,
               name = "Optimal Point", text = "Optimal Point");
sLayout = Layout(title = "Closest Point", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "y",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot([sTr1, sTr2, sTr3], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

