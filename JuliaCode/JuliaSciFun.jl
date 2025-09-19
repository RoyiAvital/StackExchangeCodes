# StackExchange Code - Julia Scientific Functions
# Set of functions for Scientific Computing.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes
# - 1.0.000     19/09/2025  Royi Avital RoyiAvital@yahoo.com
#   *   First release.

## Packages

# Internal

# External

## Constants & Configuration

if (!(@isdefined(isJuliaInit)) || (isJuliaInit == false))
    # Ensure the initialization happens only once
    include("./JuliaInit.jl");
end

# From `LogExpFunctions.jl` in `basicfuns.jl`
@inline _logistic_bounds( x :: Float16 ) = (Float16(-16.64), Float16(7.625));
@inline _logistic_bounds( x :: Float32 ) = (-103.27893f0, 16.635532f0);
@inline _logistic_bounds( x :: Float64 ) = (-744.4400719213812, 36.7368005696771);

## Functions

Logistic( valX :: T ) where {T <: AbstractFloat} = inv(one(T) + exp(-valX));

function Logistic( valX :: Union{Float16, Float32, Float64} )

    valE = exp(valX);
    lowBnd, upBnd = _logistic_bounds(valX); #<! Lower / Upper bounds
    
    return valX < lowBnd ? zero(valX) : valX > upBnd ? one(valX) : valE / (one(valX) + valE);

end

