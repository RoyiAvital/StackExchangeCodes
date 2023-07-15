# StackExchange Code - Julia Init
# Julia configuration and auxiliary functions.
# References:
#   1.  
# Remarks:
#   1.  A
# TODO:
# 	1.  B
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     09/07/2023  Royi Avital
#   *   First release.

## Packages

# Internal

# External


## Constants & Configuration

@enum PadMode begin
    PAD_MODE_CIRCULAR
    PAD_MODE_CONSTANT
    PAD_MODE_REFLECT
    PAD_MODE_REPLICATE
    PAD_MODE_SYMMETRIC
end

@enum ConvMode begin
    CONV_MODE_FULL
    CONV_MODE_SAME
    CONV_MODE_VALID
end

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## Auxiliary Functions