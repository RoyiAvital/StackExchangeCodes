# StackExchange Mathematics Q5084488
# https://math.stackexchange.com/questions/5084488
# Robust Method to Fit an Ellipse in R².
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
# - 1.0.000     22/07/2025  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Convex;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SCS;                 #<! Seems to support more cases for Continuous optimization than ECOS
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

function GenEllipseData( majRadius :: T, minRadius :: T, centerX :: T, centerY :: T, θ :: T; numPts :: N = 100, σ :: T = T(0) ) where {T <: AbstractFloat, N <: Integer}

    # vT = 2π * rand(numPts);
    vT = collect(LinRange(T(0), T(7), numPts));
    sort!(vT);
    vX = @. centerX + majRadius * cos(vT) * cos(θ) - minRadius * sin(vT) * sin(θ);
    vY = @. centerY + majRadius * cos(vT) * sin(θ) + minRadius * sin(vT) * cos(θ);

    vX .+= σ * randn(numPts);
    vY .+= σ * randn(numPts);

    return vX, vY;
    
end

function GenEllipseDataAlgebric( vP :: Vector{T}; numPts :: N = 100, σ :: T = T(0) ) where {T <: AbstractFloat, N <: Integer}

    # `vP` -> [a, b, c, d, e, f]
    paramA = vP[1];
    paramB = vP[2];
    paramC = vP[3];
    paramD = vP[4];
    paramE = vP[5];
    paramF = vP[6];

    valDisc = paramB * paramB - T(4) * paramA * paramC; #<! Discriminant
    if valDisc ≥ zero(T)
        println("Not a valid ellipse");
    end

    valLeft   = T(2) * (paramA * paramE * paramE + paramC * paramD * paramD - paramB * paramD * paramE + paramF * valDisc);
    paramAC   = paramA + paramC;
    valSqrt   = sqrt(((paramA - paramC) ^ 2) + paramB * paramB);

    majRadius = -sqrt(valLeft * (paramAC + valSqrt)) / valDisc;
    minRadius = -sqrt(valLeft * (paramAC - valSqrt)) / valDisc;
    centerX   = (T(2) * paramC * paramD - paramB * paramE) / valDisc;
    centerY   = (T(2) * paramA * paramE - paramB * paramD) / valDisc;
    θ         = T(0.5) * atan(-paramB, paramC - paramA);

    return GenEllipseData(majRadius, minRadius, centerX, centerY, θ; numPts = numPts, σ = σ);
    
end

function AddOutliers( vX :: Vector{T}, vY :: Vector{T}; outRatio = 0.25 ) where {T <: AbstractFloat}

    numSamples = length(vX);
    vOutIdx    = rand(1:numSamples, Int(round(outRatio * numSamples)));

    vX[vOutIdx] .+= 0.5;
    vY[vOutIdx] .+= 0.5;

    return vX, vY;

end

function GenMatD( vX :: Vector{T}, vY :: Vector{T} ) where {T <: AbstractFloat}

    mD = [(vX .^ 2) (vX .* vY) (vY .^ 2) vX vY ones(length(vX))];

    return mD;
    
end

function CVXSolver( mD :: Matrix{T} ) where {T <: AbstractFloat}

    vP = Variable(6); #<! [a, b, c, d, e, f]
    
    # Problem is formulated into SDP (Solvers: SCS, Clarabel, COSMO)
    # sConvProb = minimize( Convex.sumsquares(mD * vP), [(vP[1] + vP[3]) == 1, isposdef([vP[1] (vP[2] / T(2)); (vP[2] / T(2)) vP[3]]) ] );
    sConvProb = minimize( Convex.norm_1(mD * vP), [(vP[1] + vP[3]) == 1, isposdef([vP[1] (vP[2] / T(2)); (vP[2] / T(2)) vP[3]]) ] );
    Convex.solve!(sConvProb, SCS.Optimizer; silent = true);
    
    return vec(vP.value);

end

function SolvePDHG( mD :: Matrix{T}; numItr :: N = 950, ρ :: T = T(0.001), γ :: T = T(0.001) ) where {T <: AbstractFloat, N <: Integer}
    # Solve || D q ||_1 s.t. A(q) ∈ S₊ⁿ, Tr(A(q)) == 1
    # Using Chambolle Pock method
    # `γ` - The Prox coefficient for || D q ||_2^2.
    # `γ` - Regularization (Assists with conditioning `mK`), must be > 0
    # `γ` - Lower values seems to bring higher accuracy.

    # The mode:
    # \min_x f(p) + g(D p) = Iₛ(p) + || D p ||₁
    # Where Iₛ(q) is the indicator over the set A(q) ∈ S₊ⁿ, Tr(A(q)) == 1.
    
    c1 = mean(mD[:, 4]);
    c2 = mean(mD[:, 5]);
    r2 = var(mD[:, 4]) + var(mD[:, 5]);

    hProxF( vY, λ )  = ProjectP(vY);
    hProxGꜛ( vY, λ ) = ProjectL∞Ball(vY);

    # Smart initialization
    vP = [T(0.5), T(0.5), T(0), -c1, -c2, (c1 ^ 2 + c2 ^ 2 - r2) / T(2)];
    vP̄ = copy(vP);
    vQ = mD * vP; #<! Dual
    vP¹ = copy(vP); #<! Buffer to keep previous iteration

    vQQ = copy(vQ); #<! Buffer 
    vPP = copy(vP); #<! Buffer 

    for ii ∈ 1:numItr
        copy!(vQQ, vQ);
        mul!(vQQ, mD, vP̄, ρ, one(T));
        # vQ   = hProxGꜛ(vQQ, ρ);
        vQ .= clamp.(vQQ, -one(T), one(T));
        # vQ   = hProxGꜛ(vQ + ρ * mD * vP̄, ρ);
        vP¹ .= vP; #<! Previous step
        copy!(vPP, vP);
        mul!(vPP, mD', vQ, -γ, one(T));
        vP   = hProxF(vPP, γ);
        # vP   = hProxF(vP - γ * mD' * vQ, γ);
        vP̄  .= T(2) .* vP .- vP¹; 
    end

    return vP;

end

function ProjectP( vP :: Vector{T} ) where {T <: AbstractFloat}
    # 1. Extracting matrix A of the vector of parameters.
    # 2. Projecting matrix A onto the set of SPSD matrices with unit trace.
    # 3. Collecting the updated values of A into teh vector of parameters.
    
    vQ = copy(vP);
    valFctr = T(2);
    
    mA = [vP[1] (vP[2] / valFctr); (vP[2] / valFctr) vP[3]];
    mA = ProjectSPDUnitTr(mA);

    vQ[1] = mA[1];
    vQ[2] = valFctr * mA[2];
    vQ[3] = mA[4];

    return vQ;

end

function ProjectSPDUnitTr( mY :: Matrix{T} ) where {T <: AbstractFloat}
    # \arg \min_{X} 0.5 * || X - Y||_2^2 s.t. Tr(X) = 1, X is SPSD
    # Y is assumed ot be symmetric
    
    # `mY` Assumed ot be symmetric
    sF = eigen(mY); #<! Eigen decomposition

    vD = ProjectSimplex(sF.values);
    # mX = (sF.vectors .* vD') * sF.vectors';
    # mX = sF.vectors * (vD .* sF.vectors');
    mX = sF.vectors * Diagonal(vD) * sF.vectors';
    mX = T(0.5) * (mX' + mX); #<! Ensure symmetry

    return mX;

end

function ProjectSimplex( vY :: AbstractVector{T}; ballRadius :: T = T(1) ) where {T <: AbstractFloat}
    # See http://arxiv.org/abs/1101.6081
    # Sorting vector in descending order, hence looping forward (Paper loops backward)
    # TODO: Make it the go to Projection onto simplex (In `JuliaProxOperators.jl`).
    
    numElements = length(vY);
    
    vX   = sort(vY; rev = true); #<! `vX` is a sorted **copy** of `vY`
    sumT = zero(T); #<! Cumulative sum
    tᵢ   = zero(T); #<! Running sum
    isT  = false; #<! Is ̂t found before loop ends

    for ii = 1:(numElements - 1)
        sumT = sumT + vX[ii];
        tᵢ   = (sumT - ballRadius) / T(ii);
        if tᵢ >= vX[ii + 1]
            isT = true;
            break;
        end
    end

    if !(isT)
        tᵢ = (sumT + vX[numElements] - ballRadius) / T(numElements);
    end

    # The value of tᵢ is ̂t in the paper
    @. vX = max(vY - tᵢ, T(0)); #<! Subtract from the unsorted array

    return vX;

end

function ProjectL∞Ball( vY :: AbstractVector{T}; ballRadius :: T = T(1.0) ) where {T <: AbstractFloat}
    
    vX = clamp.(vY, -ballRadius, ballRadius);

    return vX;

end


## Parameters

# Data
numSamples = 100;

# Ellipse parameters
majRadius = 5.0;
minRadius = 3.0;
centerX   = 2.0;
centerY   = -1.0;
θ         = π / 6.0;

# Noise
σ = 0.0;


## Load / Generate Data

vX, vY = GenEllipseData(majRadius, minRadius, centerX, centerY, θ; σ = σ);
vX, vY = AddOutliers(vX, vY);
mD     = GenMatD(vX, vY);

dSolvers = Dict();


## Analysis
# The Model: || D q ||_1 subject to A(q) ∈ S₊ⁿ, Tr(A(q)) == 1

# DCP Solver
methodName = "Convex.jl"

vPRef = CVXSolver(mD);
# vPRef = JuMPSolver(mD);
vXX, vYY = GenEllipseDataAlgebric(vPRef; numPts = 1_000);

# dSolvers[methodName] = hObjFun(vXRef) * ones(numIterations);
# optVal = hObjFun(vXRef);

# ADMM
methodName = "Primal Dual Hybrid Gradient";

vP = SolvePDHG(mD; numItr = 50_000, ρ = 0.005, γ = 0.005);
vXX, vYY = GenEllipseDataAlgebric(vP; numPts = 1_000);

println(norm(vP - vPRef, Inf))

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];


## Display Results

figureIdx += 1;

sTr1 = scatter(; x = vX, y = vY, mode = "markers", 
               marker_size = 7,
               name = "Samples", text = "Samples");
sTr2 = scatter(; x = vXX, y = vYY, mode = "lines", 
               line_width = 2.75,              
               name = "Estimated Ellipse", text = "Estimated Ellipse");
sLayout = Layout(title = "Ellipse Fit", width = 600, height = 600, 
                 xaxis_title = "x", yaxis_title = "y",
                 hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0),
                 legend = attr(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99));

hP = Plot([sTr1, sTr2], sLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme; width = hP.layout[:width], height = hP.layout[:height]);
end

# Run Time Analysis
runTime = @belapsed CVXSolver(mD) seconds = 2;
resAnalysis = @sprintf("The Convex.jl (SCS) solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);

runTime = @belapsed SolvePDHG(mD) seconds = 2;
resAnalysis = @sprintf("The ADMM Method solution run time: %0.5f [Sec]", runTime);
println(resAnalysis);



