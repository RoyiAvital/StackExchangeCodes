# StackExchange Code - Julia Visualization
# Set of functions for data visualization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  Make a single `DisplayImage()` for `UInt8`.  
#       All others should convert `UInt8` to and use it.
# Release Notes
# - 1.5.000    29/04/2025  Royi Avital RoyiAvital@yahoo.com
#   *   Added `PlotDft()`.
# - 1.4.000    13/02/2025  Royi Avital RoyiAvital@yahoo.com
#   *   Added `PlotLine()`.
# - 1.3.000    14/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added support to for mask (`BitMatrix` / `Matrix{Bool}`) in `DisplayImage()`.
# - 1.2.001    08/09/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Verifying the initialization happens only once.
# - 1.2.000    03/07/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added support to display `UIntX` images.
# - 1.1.000    01/07/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Added `PlotSparseMat()`.
# - 1.0.001     29/06/2024  Royi Avital RoyiAvital@yahoo.com
#   *   Made `T` a float type.
#   *   Fixed dynamic range for `heatmap()`.
# - 1.0.000     24/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal
using SparseArrays;

# External
using PlotlyJS;

## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

vPlotlyDefColors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]; #<! See https://stackoverflow.com/questions/40673490

## Functions

function DisplayImage(mI :: Matrix{T}; tuImgSize :: Tuple{N, N} = size(mI), titleStr :: String = "" ) where {T <: AbstractFloat, N <: Integer}
    # Displays a grayscale image in the range [0, 1]
    
    oTr1 = heatmap(z = UInt8.(round.(T(255) * clamp.(mI, zero(T), one(T))))[end:-1:1, :], showscale = false, colorscale = "Greys", zmin = UInt8(0), zmax = UInt8(255));
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function DisplayImage(mI :: Array{T, 3}; tuImgSize :: Tuple{N, N} = size(mI)[1:2], titleStr :: String = "" ) where {T <: AbstractFloat, N <: Integer}
    # Displays an RGB image in the range [0, 1]
    
    oTr1 = image(z = permutedims(UInt8.(round.(T(255) * clamp.(mI, zero(T), one(T)))), (3, 2, 1)), colormodel = "rgb");
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function DisplayImage(mI :: Matrix{T}; tuImgSize :: Tuple{N, N} = size(mI), titleStr :: String = "" ) where {T <: Unsigned, N <: Integer}
    # Displays a UInt grayscale image

    maxVal = typemax(T);
    sclRatio = maxVal / 255.0; #<! To UInt8
    
    oTr1 = heatmap(z = UInt8.(floor.(mI ./ sclRatio))[end:-1:1, :], showscale = false, colorscale = "Greys", zmin = UInt8(0), zmax = UInt8(255));
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function DisplayImage(mI :: Array{T, 3}; tuImgSize :: Tuple{N, N} = size(mI)[1:2], titleStr :: String = "" ) where {T <: Unsigned, N <: Integer}
    # Displays a UInt RGB image

    maxVal = typemax(T);
    sclRatio = maxVal / 255.0; #<! To UInt8
    
    oTr1 = image(z = permutedims(UInt8.(floor.(mI ./ sclRatio)), (3, 2, 1)), colormodel = "rgb");
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function DisplayImage(mI :: BitMatrix; tuImgSize :: Tuple{N, N} = size(mI), titleStr :: String = "" ) where {N <: Integer}
    # Displays a Boolean image (Mask)
    
    return DisplayImage(UInt8(255) .* UInt8.(mI); tuImgSize = tuImgSize, titleStr = titleStr);

end

function DisplayImage(mI :: Matrix{Bool}; tuImgSize :: Tuple{N, N} = size(mI), titleStr :: String = "" ) where {N <: Integer}
    # Displays a Boolean image (Mask)
    
    return DisplayImage(UInt8(255) .* UInt8.(mI); tuImgSize = tuImgSize, titleStr = titleStr);

end

function PlotMatrix( vX :: Vector, vY :: Vector, mI :: Matrix{T}; titleStr :: String = "", colorMap :: String = "RdBu" ) where {T <: Real}
    # Displays a Matrix
    # `colorMap`: Blackbody,Bluered,Blues,Cividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd
    # https://plotly.com/python/builtin-colorscales
    # https://community.plotly.com/t/11730
    
    oTr1 = heatmap(x = vX, y = vY, z = mI, showscale = false, colorscale = colorMap);
    oLayout = Layout(title = titleStr, width = size(mI, 2) + 100, height = size(mI, 1) + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function PlotMatrix( mI :: Matrix{T}; titleStr :: String = "", colorMap :: String = "RdBu" ) where {T <: Real}
    # Displays a Matrix
    # `colorMap`: Blackbody,Bluered,Blues,Cividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd
    # https://plotly.com/python/builtin-colorscales
    # https://community.plotly.com/t/11730
                
    oTr1 = heatmap(z = mI, showscale = false, colorscale = colorMap);
    oLayout = Layout(title = titleStr, width = size(mI, 2) + 100, height = size(mI, 1) + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = Plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function PlotSparseMat( mM :: AbstractSparseMatrix )
    # Works with up to ~100K elements

    numRows = size(mM, 1);
    numCols = size(mM, 2);
    vI, vJ, vV = findnz(mM);

    oTr = scattergl(x = vJ, y = vI, mode = "markers",
                  text = "Sparse Matrix", name = "Sparse Matrix", marker = attr(size = 5.0))
    oLayout = Layout(title = "Sparse Matrix Pattern", width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "Column", yaxis_title = "Row",
                  yaxis_range = [numRows, 1], xaxis_range = [1, numCols]);
    
    hP = Plot([oTr], oLayout);

    return hP; #<! display(hP);

end

function PlotLine( vX :: Vector{T1}, vY :: VecOrMat{T2}; plotTitle :: String = "", signalName :: String = "", vSigNames :: Vector{String} = [""], xTitle :: String = "x", yTitle :: String = "y", plotMode = "lines" ) where {T1 <: Real, T2 <: Real}

    numLines = size(vY, 2);
    
    vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, numLines);
    sigName = signalName;
    if length(vSigNames) == 1
        vSigNames = repeat(vSigNames, numLines);
    end

    for ii ∈ 1:numLines
        if (numLines > 1)
            sigName = vSigNames[ii];
        end
        vTr[ii] = scatter(; x = vX, y = vY[:, ii], mode = plotMode, name = sigName);
    end

    oLayout = Layout(title = plotTitle, width = 600, height = 600, hovermode = "closest",
                  xaxis_title = xTitle, yaxis_title = yTitle);
    
    hP = Plot(vTr, oLayout);

    return hP; #<! display(hP);

end

function PlotLine( vY :: VecOrMat{T}; plotTitle :: String = "", signalName :: String = "", vSigNames :: Vector{String} = [""], xTitle = "x", yTitle = "y", plotMode = "lines" ) where {T <: Real}

    return PlotLine(T.(collect(1:size(vY, 1))), vY; plotTitle, signalName, vSigNames = vSigNames, xTitle = xTitle, yTitle = yTitle, plotMode = plotMode);

end

function PlotDft( vK :: VecOrMat{T}, samplingFrequency :: T, numSamples :: N; singleSideFlag :: Bool = true, logScaleFlag :: Bool = true, normalizeData :: Bool = true, plotTitle :: String = "DFT" ) where {T <: AbstractFloat, N <: Integer}
    # vK - Real (Absolute Value) of the DFT

    if singleSideFlag
        numFreqBins = size(vK, 1);
    else
        numFreqBins = numSamples;
    end

    # The frequency grid
    vF = LinRange(0, samplingFrequency, numSamples + 1);
    vF = vF[1:numFreqBins];

    numSignals = size(vK, 2);

    vTr = Vector{GenericTrace{Dict{Symbol, Any}}}(undef, numSignals);
    
    for ii ∈ 1:numSignals
        vTr[ii] = scatter(; x = vF, y = vK[:, ii], mode = "lines");
    end

    oLayout = Layout(title = plotTitle, width = 600, height = 600, hovermode = "closest",
                  xaxis_title = "Frequency", yaxis_title = "Amplitude");
    
    hP = Plot(vTr, oLayout);

    return hP; #<! display(hP);


end
