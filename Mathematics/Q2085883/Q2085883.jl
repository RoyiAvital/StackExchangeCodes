# StackExchange Mathematics Q2085883
# https://math.stackexchange.com/questions/2085883
# Linear Least Squares with Norm Equality Constraint.
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
# - 1.0.000     17/11/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
using Random;
# External
using LinearAlgebra;
using Optim;
using PlotlyJS;
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)
# Random.default_rng() = StableRNG(RNG_SEED); #<! Danger! This is a hack.

## General Parameters

figureIdx = 0;

exportFigures = false;

dUtfSymPx   = Dict(UInt8(0) => '🟩', UInt8(128) => '🟦', UInt8(255) => '🟥');
dUtfSymBool = Dict(false => '🟥', true => '🟩');
dUtfSymDir  = Dict(Int8(-1) => '↖', Int8(0) => '↑', Int8(1) => '↗');

## Functions


## Parameters

numRows = 3;
numCols = 5;
δ = 0;

numGridPts = 1000;

## Generate / Load Data

mA = randn(numRows, numCols);
vB = randn(numRows);

hObjFun( vX :: Vector{<: AbstractFloat} ) = 0.5 * sum(abs2, mA * vX - vB) + δ * (sum(abs2, vX) > 1);

## Analysis
sSvdFac = svd(mA);
vC = sSvdFac.U' * vB;

hD( λ :: T ) where {T <: AbstractFloat} = sSvdFac.S ./ ((sSvdFac.S .^ 2) .+ 2λ);
hF( λ :: T ) where {T <: AbstractFloat} = sSvdFac.V * (hD(λ) .* vC);
hG( λ :: T ) where {T <: AbstractFloat} = (sum(abs2, hF(λ)) - 1) ^ 2;
hS( λ :: T ) where {T <: AbstractFloat} = mA' * (mA * hF(λ) - vB) + 2λ * hF(λ);

# The function is monotonic decreasing from this point (First root) down
leftλVal = -((sSvdFac.S[end] ^ 2) / 2); # Where hD goes to Inf with this λ

# Seems to work when hF(0.0) > 1 or numRows > numCols.
sOptRes = optimize(hG, leftλVal + 0.05, 1000, Brent());
sOptRes.minimizer

# The λ path
vλ = LinRange(leftλVal + 0.05, sSvdFac.S[1], numGridPts);
vObjVal = [sum(abs2, hF(λ)) for λ in vλ];

## Display Results

figureIdx += 1;

# shapeLine = vline(sOptRes.minimizer, line_color = "green", name = "Optimal λ");
oTrace0 = scatter(x = [sOptRes.minimizer, sOptRes.minimizer], y = [minimum(vObjVal), maximum(vObjVal)], mode = "lines", text = "Optimal λ", name = "Optimal λ",
                  line = attr(width = 3.0));
oTrace1 = scatter(x = vλ, y = vObjVal, mode = "lines", text = "λ Path", name = "λ Path",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = vλ, y = ones(numGridPts), 
                  mode = "lines", text = "Norm 1", name = "Norm 1",
                  line = attr(width = 1.5, dash = "dot"));
oLayout = Layout(title = "Squared Norm of f(λ)", width = 600, height = 600, hovermode = "closest",
                 xaxis_title = "λ", yaxis_title = "Squared L2 Norm");

hP = plot([oTrace0, oTrace1, oTrace2], oLayout);
display(hP);

if (exportFigures)
    figFileNme = @sprintf("Figure%04d.png", figureIdx);
    savefig(hP, figFileNme);
end

# https://math.stackexchange.com/a/9101
# I think this will work only if size(mA, 2) > size(mA, 1) and || pinv(mA) * vB || < 1.
# F = svd(mA; full = true);
# vN = F.V[:, end];
# vXX = pinv(mA) * vB;
# α = sqrt(1 - sum(abs2, vXX));
# vX = vXX + α * vN;
# sum(abs2, vX)

# Naive Projected Gradient Descent.
# Seems to work in all cases.
# https://trungvietvu.github.io/files/MLSP19.pdf

numIterations = 10000;
α = 0.5 / ((opnorm(mA) ^ 2) + 1);

vXX = pinv(mA) * vB;
vY = vXX / sqrt(sum(abs2, vXX));
mAA = mA' * mA;
vAb = mA' * vB;
vT = fill(0.0, size(vY));

for ii ∈ 2:numIterations
    vT .= mul!(vT, mAA, vY) .- vAb;
    vY .-= α .* vT;
    vY ./= sqrt(sum(abs2, vY));
end

println(maximum(abs.(vY - hF(sOptRes.minimizer))))
println(abs(hObjFun(vY) - hObjFun(hF(sOptRes.minimizer))))
