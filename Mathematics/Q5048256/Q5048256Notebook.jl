### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ f4a4a7a4-1fa7-4705-8056-01ddf0b5b959
# Using the Local Project
begin
	using Pkg;
	Pkg.activate(Base.current_project());
end

# ╔═╡ 6850a4c1-e535-4b26-b4a9-6af1fd8562aa
# Packages
begin
# Internal
using DelimitedFiles;      #<! Read CSV
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Distributions;
using FastLapackInterface; #<! Required for Optimization
using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using SparseArrays;
using StableRNGs;
end

# ╔═╡ 15cda02d-d66f-4b4a-88ce-3f3e2f2d91b9
## Constants & Configuration
begin
RNG_SEED = 1234;

juliaCodePath = joinpath(".", "..", "..", "JuliaCode");
include(joinpath(juliaCodePath, "JuliaInit.jl"));
include(joinpath(juliaCodePath, "JuliaVisualization.jl"));
end

# ╔═╡ 32d08d5c-d1cd-4106-8407-78c7c6ac4098
# From https://discourse.julialang.org/t/76812/5
html"""
<script>
	const button = document.createElement("button")

	button.addEventListener("click", () => {
		editor_state_set(old_state => ({
			notebook: {
				...old_state.notebook,
				process_status: "no_process",
			},
		})).then(() => {
			window.requestAnimationFrame(() => {
				document.querySelector("#process_status a").click()
			})
		})
	})
	button.innerText = "Restart Notebook"

	return button
</script>
"""

# ╔═╡ 9433d1a0-06fa-11f0-2c4e-07a5faa80a6d
md"""
# Maximum Likelihood of Sum of 2 Uniform Variables

Given:

$${Z}_{i} = {X}_{i} + {Y}_{i}, \; {X}_{i} \sim \mathbb{U}_{\left[ a - r, a + r \right]}, \, {Y}_{i} \sim \mathbb{U}_{\left[ b - r, b + r \right]}$$

Estimate $r$ for $\left\{ {Z}_{i} \right\}_{i = 1}^{n}$ using the _Maximum Likelihood Estimator_.
"""

# ╔═╡ c32031d0-a6a5-416c-952d-bf49fa9088ca
## Settings

begin
exportFigures = true;

oRng = StableRNG(1234);

mutable struct FigureIdx
	idx :: Int;
end

function YieldFigIdx(F :: FigureIdx)
	F.idx += 1;
	return F.idx;
end

function YieldFigFileName(F :: FigureIdx)
	return @sprintf("Figure%04d.png", YieldFigIdx(sFigureIdx));
end

sFigureIdx = FigureIdx(0);

end

# ╔═╡ 6644718c-ce08-49c3-aa04-83a1e2a6aae7
## Functions

function PdfSumUniform( valT :: T, X :: Uniform{T}, Y :: Uniform{T} ) where {T <: AbstractFloat}
    # Analytic Solution
    #                                                  
    #               │                  │               
    #               │                  │               
    #              x│xxxxxxxxxxxxxxxxxx│x              
    #             xx│                  │xxx            
    #           xxx │                  │  xx           
    #          xx   │                  │    xx         
    #        xxx    │                  │     xxx       
    #      xxx      │                  │       xx      
    #     xx        │                  │        xxx    
    # ───x──────────┼──────────────────┼──────────x─   
    #   a+c        c+b                a+d         b+d  
    #                                                  
    # Assuming X ~ U[a, b], Y ~ U[c, d] with (d - c) >= (b - a).
    
    # Ensuring the assumption for XX and YY.
    suppX = maximum(X) - minimum(X);
    suppY = maximum(Y) - minimum(Y);

    if suppX > suppY
        XX = Y;
        YY = X;
        suppX, suppY = suppY, suppX;
    else
        XX = X;
        YY = Y;
    end

    valMin = minimum(X) + minimum(Y); #<! a + c
    valMax = maximum(X) + maximum(Y); #<! b + d

    if (valT > valMax) || (valT < valMin)
        return zero(T);
    end

    valA = minimum(X);
    valB = maximum(X);
    valC = minimum(Y);
    valD = maximum(Y);

    valCenter = inv(suppY) #<! Should be: `inv(suppX) * inv(suppY) * suppX;`
    valSlope  = valCenter / suppX;

    if valT <= (valC + valB)
        valOut = valSlope * (valT - valMin);
    elseif valT <= (valA + valD)
        valOut = valCenter;
    else #<! `(valT <= (valB + valD))`
        valOut = valSlope * (valMax - valT);
    end
    
    return valOut;
    
end


# ╔═╡ 9db16a21-78ee-4ce5-84f6-7cd5081db994
## Functions

function CalLogLikelihood( vZ :: Vector{T}, paramR :: T, valA :: T, valB :: T ) where {T <: AbstractFloat}

    numSamples = length(vZ);

    logLik = zero(T);

    for ii ∈ 1:numSamples
        logLik += log(T(2) * paramR - abs(vZ[ii] - valA - valB));
    end

    logLik -= T(2) * numSamples * log(T(2) * paramR);
    
    return logLik;
    
end

# ╔═╡ b1b36cd1-45a1-4326-aff4-e89275ff1d61
md"""
Given that the length of the support of ${x}_{i}$ and ${y}_{i}$ are the same the PDF of ${z}_{i}$ is easy:

$${f}_{z; r} \left( z \right) =  \begin{cases}
0 & \text{if} \; z < a + b - 2R \\
\frac{z - a - b + 2R}{4 {R}^{2}} & \text{if} \; a + b - 2R \leq z \leq a + b \\
\frac{a + b + 2R - z}{4 {R}^{2}} & \text{if} \; a + b \leq z \leq a + b + 2R \\
0 & \text{if} \; z > a + b + 2R \\
\end{cases}$$

Which is equivalent to:

$${f}_{z; r} \left( z \right) =  \begin{cases}
0 & \text{if} \; z < a + b - 2R \\
\frac{ 2R - \left| z - a - b \right|}{4 {R}^{2}} & \text{if} \; a + b - 2R \leq z \leq a + b + 2R \\
0 & \text{if} \; z > a + b + 2R \\
\end{cases}$$

"""

# ╔═╡ bb57860f-6210-4ee5-b201-70c47d8fca3b
## Parameters

begin
	
numSamples = 500;

paramA = 3.0;
paramB = 5.0;
paramR = 1.0;

# Solver
numGridPts = 20_001;

end;

# ╔═╡ c3b837d6-2364-410d-bdb8-cfa220f5beb1
## Load / Generate Data
begin
X = Uniform(paramA - paramR, paramA + paramR);
Y = Uniform(paramB - paramR, paramB + paramR);
hPdfZ(valT :: T) where {T <: AbstractFloat} = PdfSumUniform(valT, X, Y);
minG = min(paramA - paramR - 1.0 , paramA + paramB - 2paramR - 1.0);
maxG = max(paramB + paramR + 1.0, paramA + paramB + 2paramR + 1.0);
vG = LinRange(minG, maxG, numGridPts);
pdfX = pdf(X, vG);
pdfY = pdf(Y, vG);
pdfZ = hPdfZ.(vG);

# Samples
vX = rand(oRng, X, numSamples);
vY = rand(oRng, Y, numSamples);
vZ = vX + vY;
end;

# ╔═╡ e5399666-e8b3-4b72-91ce-54ccb0eaa407
## Analysis

begin

# The minimum value of R must match the data
minR = ceil(maximum(abs.(vZ .- paramA .- paramB)) / 2.0; digits = 3);
vRGrid = LinRange(minR, minR + 0.5, 1_000);
vR = [CalLogLikelihood(vZ, valR, paramA, paramB) for valR ∈ vRGrid];

end;

# ╔═╡ d92d74dd-253a-4e54-987a-0f2a4fc2374a
## Display Analysis

begin

mA = [pdfX;; pdfY;; pdfZ];

hP1 = PlotLine(collect(vG), mA; plotTitle = "The PDF's", vSigNames = ["X", "Y", "Z"]);
display(hP1);

if (exportFigures)
    savefig(hP1, YieldFigFileName(sFigureIdx));
end

hP1

end

# ╔═╡ 86456d48-fdfc-46dd-be67-e2994145efa8
## Display Analysis

begin

oTrace1 = scatter(x = vRGrid, y = vR, mode = "lines", text = "Log Likelihood", name = "Log Likelihood",
                  line = attr(width = 3.0));
oTrace2 = scatter(x = [vRGrid[argmax(vR)]], y = [maximum(vR)], 
                  mode = "markers", text = "Maximum Value", name = "Maximum Value",
                  marker = attr(size = 12, color = "r"));

oLayout = Layout(title = "The Log Likelihood Function", width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "R", yaxis_title = "L(z; R)");
 hP2 = Plot([oTrace1, oTrace2], oLayout);
 display(hP2);

if (exportFigures)
    savefig(hP2, YieldFigFileName(sFigureIdx));
end

hP2

end

# ╔═╡ Cell order:
# ╟─32d08d5c-d1cd-4106-8407-78c7c6ac4098
# ╠═f4a4a7a4-1fa7-4705-8056-01ddf0b5b959
# ╟─9433d1a0-06fa-11f0-2c4e-07a5faa80a6d
# ╠═6850a4c1-e535-4b26-b4a9-6af1fd8562aa
# ╠═15cda02d-d66f-4b4a-88ce-3f3e2f2d91b9
# ╠═c32031d0-a6a5-416c-952d-bf49fa9088ca
# ╠═6644718c-ce08-49c3-aa04-83a1e2a6aae7
# ╠═9db16a21-78ee-4ce5-84f6-7cd5081db994
# ╟─b1b36cd1-45a1-4326-aff4-e89275ff1d61
# ╠═bb57860f-6210-4ee5-b201-70c47d8fca3b
# ╠═c3b837d6-2364-410d-bdb8-cfa220f5beb1
# ╠═e5399666-e8b3-4b72-91ce-54ccb0eaa407
# ╠═d92d74dd-253a-4e54-987a-0f2a4fc2374a
# ╠═86456d48-fdfc-46dd-be67-e2994145efa8
