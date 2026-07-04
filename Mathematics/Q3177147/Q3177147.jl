# StackOverflow Q3177147
# https://math.stackexchange.com/questions/3177147
# Minimize L1 Norm || x a - b ||_1 without Linear Programming
# References:
#   1.  
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   3. 
# TODO:
# 	1.  C
# Release Notes
# - 1.0.000     27/01/2024  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using Convex;
using Optim;
using PlotlyJS;
using SCS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
# include(joinpath(juliaCodePath, "JuliaOptimization.jl"));

## General Parameters

figureIdx       = 0;
exportFigures   = false;

oRng = StableRNG(123);

## Functions

function ObjFun( x :: T, vA :: Vector{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    objVal = zero(T);
    @simd for ii in 1:length(vA)
        @fastmath objVal += abs(x * vA[ii] - vB[ii]);
    end

    return objVal;
    
end

function ∂F( x :: T, vA :: Vector{T}, vB :: Vector{T} ) where {T <: AbstractFloat}

    ∂Val = zero(T);
    @simd for ii in 1:length(vA)
        @fastmath ∂Val += vA[ii] * sign(x * vA[ii] - vB[ii]);
    end

    return ∂Val;
    
end

function LinSlopeSign( ii :: N, vX :: Vector{T} ) where {T <: AbstractFloat, N <: Integer}

    return sign(vX[ii + 1] - vX[ii]);
    
end

function ProxF( valY :: T, vA :: Vector{T}, vB :: Vector{T}, λ :: T ) where {T <: AbstractFloat}
    # The (Approximated) Proximal Operator of the function f(x) = || x a - b ||_1
    # It is assproximated as it assumes $\boldsymol{a} \boldsymbol{a}^{T} = α I$, which is not true.
    # It basically treats the problem as a separable Huber Loss functions. 
    # In this iterated case it actually solves teh enevlop, the Huber Loss with δ = λ, which is the Moreau Envelope of f(x) with parameter λ.
    # To solve the case of L1 norm, it should be iterated until convergence with λ → 0.

    α     = sum(abs2, vA);
    μ     = inv(α);
    vT    = valY .* vA .- vB;
    vT   .= (max.(abs.(vT) .- α * λ, zero(T)) .* sign.(vT)) .- (valY .* vA) .+ vB;
    valY += μ * dot(vA, vT);

    return valY;

end

function BinarySearch( vX :: Vector{T}, keyVal :: T; leftIdx :: N = 1, rightIdx :: N = length(vX), hF = identity ) where {T <: Number, N <: Integer}
    # Should match searchsortedfirst() without applying `by` on the input as well.

    leftIdx > rightIdx && return -1;
    leftIdx == rightIdx && return leftIdx;

    while (leftIdx < rightIdx)
        midIdx  = (leftIdx + rightIdx) ÷ 2;
        # midVal  =  hF(vX[midIdx]);
        midVal  =  hF(midIdx);

        midVal == keyVal && return midIdx;

        # @printf("leftIdx = %d, rightIdx = %d, midIdx = %d, midVal = %0.6f\n", leftIdx, rightIdx, midIdx, midVal);
        if (midVal < keyVal)
            leftIdx = midIdx + 1; #<! Using `midIdx + 0` instead of `midIdx + 1` to keep the interval of change of sign
        else
            rightIdx = midIdx - 1;
        end
    end

    return leftIdx;
    
end


## Parameters

# Data
numGridPts  = 5000;
numElements = 10;

# Solvers
numIterations = 10_000;
ε             = 1e-8;

## Generate / Load Data
vA = randn(oRng, numElements);
vB = randn(oRng, numElements);

hObjFun( x ) = ObjFun(x, vA, vB);
h∂F( x )     = ∂F(x, vA, vB);

dSolvers = Dict();

vC = vB ./ vA;
sort!(vC);
# Add element to the end
vC̄ = vcat(vC, vC[numElements] + 1.0);
vG = LinRange(vC[1] - 1.0, vC[numElements] + 1.0, numGridPts);

hLinSlopeSign( ii ) = LinSlopeSign(ii, hObjFun.(vC̄));

# Display Data

figureIdx += 1;

vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, 2);

vTr[1] = scatter(x = vG, y = hObjFun.(vG), 
               mode = "lines", text = "Objective Function", name = "Objective Function", line = attr(width = 3.0));
vTr[2] = scatter(x = vC, y = hObjFun.(vC), 
               mode = "markers", text = "Control Points", name = "Control Points", line = attr(width = 3.0));
oLayout = Layout(title = "Objective Function", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "x", yaxis_title = "Objective Value");

hP = plot(vTr, oLayout);
display(hP);

## Analysis

# DCP Solver
methodName = "DCP Solver";

valX0 = Variable(1);
sConvProb = minimize(norm(valX0 * vA - vB, 1));
solve!(sConvProb, SCS.Optimizer; silent_solver = true);
valX0 = valX0.value
optVal = sConvProb.optval;

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX0, optVal);

# Piece Wise Linear analysis (https://math.stackexchange.com/a/3177492)
methodName = "Piece Wise Linear";

# Find the `k` such that [f(x_k), f(x_{k+1})] is the first segment with a non negative slope
kk   = BinarySearch(vC, 0.0, hF = hLinSlopeSign); #<! Zero or positive
valX = vC[kk];

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX, hObjFun(valX));

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# Fixed Point Prox (https://math.stackexchange.com/a/3779922)
methodName = "Fixed Point Prox"; #<! This implementation does not work

λ        = 1.0;
valX     = 0.0;
valXPrev = 0.0;

for ii = 1:200_000
    global valX;
    global valXPrev;

    valXPrev = valX;
    λᵢ = λ / ii; #<! Decreasing λ to approach the L1 norm

    valX = ProxF(valX, vA, vB, λᵢ);

    if (abs(valX - valXPrev) <= ε) && (ii > 100)
        # @printf("Converged after %d iterations\n", ii);
        break;
    end
end

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX, hObjFun(valX));

# Fixed Point Prox (https://math.stackexchange.com/a/3779922)
methodName = "Primal Dual Hybrid Gradient"; #<! This implementation does not work
# Solves \arg \min_x || x a - b ||_1 using Chambolle Pock algorithm.
# The problem becomes \arg \min_x max_p p^T (x a - b) + δ_{||p||_\infty \le 1}(p).
# Verify convergence both of the primal and dual variables.

σ = sqrt(inv(sum(abs2, vA) + 1e-1));
τ = σ;
θ = 1.0;
vP = zeros(numElements);
vPPrev = zeros(numElements);

valX        = 0.0;
valXPrev    = 0.0;
valX̄        = 0.0;
λ           = sum(abs2, vA);
μ           = 1 / λ;
vT          = zeros(numElements); #<! Buffer

for ii = 1:200_000
    global valX;
    global valXPrev;
    global valX̄;

    valXPrev = valX;
    copy!(vPPrev, vP);
    
    vP .= clamp.(vP .+ σ .* (valX̄ .* vA .- vB), -1.0, 1.0);
    valX = valX - τ * dot(vA, vP);
    valX̄ = valX + θ * (valX - valXPrev);

    if (abs(valX - valXPrev) <= ε) && (norm(vP - vPPrev) <= ε)
        # @printf("Converged after %d iterations\n", ii);
        break;
    end
end

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX, hObjFun(valX));

methodName = "Weighted Median"; #<! This implementation does not work

vZ = vB ./ vA; #<! Assume `vA` is non zero
vI = sortperm(vZ); #<! Ascending order
vZ = vZ[vI];
vW = abs.(vA[vI]);
idxK = findfirst(cumsum(vW) .>= sum(vW) / 2);
valX = vZ[idxK];

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX, hObjFun(valX));

# dSolvers[methodName] = [hObjFun(mX[:, ii]) for ii ∈ 1:size(mX, 2)];

# Univarite Optimization
methodName = "Univariate Optimization"; #<! This implementation does not work
sOpt = optimize(hObjFun, vC[1], vC[numElements], Brent());
valX = Optim.minimizer(sOpt);

@printf("%s: x = %0.6f, f(x) = %0.6f\n", methodName, valX, hObjFun(valX));


## Display Results

# figureIdx += 1;

# vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, length(dSolvers));

# # shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal Value");
# for (ii, methodName) in enumerate(keys(dSolvers))
#     vTr[ii] = scatter(x = 1:numIterations, y = 20 * log10.(abs.(dSolvers[methodName] .- optVal) ./ abs(optVal)), 
#                mode = "lines", text = methodName, name = methodName, line = attr(width = 3.0));
# end
# oLayout = Layout(title = "Objective Function, Condition Number = $(@sprintf("%0.3f", cond(mA)))", width = 600, height = 600, hovermode = "closest",
#                  xaxis_title = "Iteration", yaxis_title = raw"$$\frac{ \left| {f}^{\star} - {f}_{i} \right| }{ \left| {f}^{\star} \right| }$ [dB]$");

# hP = plot(vTr, oLayout);
# display(hP);

# if (exportFigures)
#     figFileNme = @sprintf("Figure%04d.png", figureIdx);
#     savefig(hP, figFileNme);
# end