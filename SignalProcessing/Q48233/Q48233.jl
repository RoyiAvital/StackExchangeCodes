# StackExchange Signal Processing Q48233
# https://dsp.stackexchange.com/questions/48233
# Curve Fit of Step Function with Boundary on the 2nd Derivative
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
# - 1.0.000     02/08/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using Optim;
using PlotlyJS;

## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = false;

## Functions

function Sigmoid( x :: T; α :: T = 1 ) where {T <: AbstractFloat}

    return inv(exp(-α * x) + one(T));

end

function ∇Sigmoid( x :: T; α :: T = 1 ) where {T <: AbstractFloat}

    σ = Sigmoid(x; α = α);
    return α * σ * (1 - σ);

end

function ∇²Sigmoid( x :: T; α :: T = 1 ) where {T <: AbstractFloat}

    σ = Sigmoid(x; α = α);
    return α * α * σ * (1 - σ) * (2 - σ);

end

function ObjFun( α :: T, ε :: T, vX :: AbstractVector{T} ) where {T <: AbstractFloat}

    return abs2(maximum(∇²Sigmoid.(vX; α = α)) - ε);

end


## Parameters

numSamples    = 2000;
supportRadius = 10;

vα = [1.0, 2.0, 3.0];
vε = [0.5, 0.75];

## Load / Generate Data

numσ = length(vα);
numε = length(vε);

vX = LinRange(-supportRadius, supportRadius, numSamples);
vY = ones(numSamples);
vY[vX .< 0] .= 0;
mσ = zeros(numSamples, numσ);

for ii in 1:numσ
    mσ[:, ii] = Sigmoid.(vX; α = vα[ii]);
end

## Analysis

# Find Optimal α per Boundary

vαOpt = zeros(numε);

for ii in 1:numε
    sOptRes = optimize(α -> ObjFun(α, vε[ii], vX), 0.001, 100, Brent());
    vαOpt[ii] = sOptRes.minimizer;
end

## Display Results

# Display the function variants
vTr = [scatter(;x = vX, y = vY, mode = "lines", name = "Heavy Step")];
for ii in 1:numσ
    nameStr = @sprintf "Sigmoid α = %0.2f" vα[ii];
    push!(vTr, scatter(;x = vX, y = mσ[:, ii], mode = "lines", name = nameStr))
end
oLyt = Layout(;title = "Function Variants");
hP = plot(vTr, oLyt);

display(hP);

figureIdx += 1;

if exportFigures
    fileName = @sprintf "Figure%04i.png" figureIdx;
    savefig(hP, fileName);
end

# Display the results of optimization
vTr = [scatter(;x = vX, y = vY, mode = "lines", name = "Heavy Step")];
for ii in 1:numε
    ε = vε[ii];
    α = vαOpt[ii];
    nameStr = @sprintf "Sigmoid α = %0.2f, ε = %0.2f" α ε;
    push!(vTr, scatter(;x = vX, y = Sigmoid.(vX; α = α), mode = "lines", name = nameStr))
end
oLyt = Layout(;title = "Bounded 2nd Derivative");
hP = plot(vTr, oLyt);

display(hP);

figureIdx += 1;

if exportFigures
    fileName = @sprintf "Figure%04i.png" figureIdx;
    savefig(hP, fileName);
end
