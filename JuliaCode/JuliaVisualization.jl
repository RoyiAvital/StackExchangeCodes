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




