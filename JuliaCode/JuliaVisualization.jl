# StackExchange Code - Julia Visualization
# Set of functions for data visualization.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.0.000     24/11/2023  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External
using PlotlyJS;

## Constants & Configuration

include("./JuliaInit.jl");

## Functions

function DisplayImage(mI :: Matrix{T}; tuImgSize :: Tuple{N, N} = size(mI), titleStr :: String = "" ) where {T, N <: Integer}
    # Displays a grayscale image in the range [0, 1]
    
    oTr1 = heatmap(z = UInt8.(round.(255 * mI))[end:-1:1, :], showscale = false, colorscale = "Greys");
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end

function DisplayImage(mI :: Array{T, 3}; tuImgSize :: Tuple{N, N} = size(mI)[1:2], titleStr :: String = "" ) where {T, N <: Integer}
    # Displays an RGB image in the range [0, 1]
    
    oTr1 = image(z = permutedims(UInt8.(round.(255 * mI)), (3, 2, 1)), colormodel = "rgb");
    oLayout = Layout(title = titleStr, width = tuImgSize[2] + 100, height = tuImgSize[1] + 100, 
                hovermode = "closest", margin = attr(l = 50, r = 50, b = 50, t = 50, pad = 0));
                
    hP = plot([oTr1], oLayout);
    
    return hP; #<! display(hP);

end




