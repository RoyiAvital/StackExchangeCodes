# StackExchange Signal Processing Q88303
# https://dsp.stackexchange.com/questions/88303
# Detect Longest Vertical Lines (Edges) in an Image
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
# - 1.0.000     24/06/2023  Royi Avital
#   *   First release.

## Packages

# Internal
using Printf;
# External
using ColorTypes;
using FileIO;
import FreeType;
using UnicodePlots;

## Constants & Configuration

# Display UIntx numbers as integers
Base.show(io::IO, x::T) where {T<:Union{UInt, UInt128, UInt64, UInt32, UInt16, UInt8}} = Base.print(io, x)

## General Parameters

figureIdx = 0;

exportFigures = true;

dUtfSymPx   = Dict(UInt8(0) => '🟩', UInt8(128) => '🟦', UInt8(255) => '🟥');
dUtfSymBool = Dict(false => '🟥', true => '🟩');
dUtfSymDir  = Dict(Int8(-1) => '↖', Int8(0) => '↑', Int8(1) => '↗');

## Functions

function ConvertPackedImg(mI :: Matrix{T}) where {T}
    
    numRows, numCols = size(mI);
    numChannels = length(T);
    dataType = eltype(T).types[1]

    if numChannels > 1
        mO = Array{dataType, 3}(undef, numRows, numCols, numChannels);
        for ii ∈ 1:numRows, jj ∈ 1:numCols
            for kk ∈ 1:numChannels
                mO[ii, jj, kk] = getfield(getfield(mI[ii, jj], kk), 1); #<! According to ColorTypes data always in order RGBA
            end
        end
    else
        mO = Matrix{dataType}(undef, numRows, numCols);
        for ii ∈ 1:numRows, jj ∈ 1:numCols
            mO[ii, jj] = getfield(mI[ii, jj], 1);
        end
    end

    return mO;

end


function Test(mI :: Matrix{<: Color{T, N}}) where {T, N}
    
    numRows, numCols = size(mI);
    numChannels = N;
    dataType = T.types[1];

    mO = Array{dataType, 3}(undef, numRows, numCols, numChannels);
    for ii ∈ 1:numRows, jj ∈ 1:numCols
        for kk ∈ 1:numChannels
            mO[ii, jj, kk] = getfield(getfield(mI[ii, jj], kk), 1); #<! According to ColorTypes data always in order RGBA
        end
    end

    return mO;

end

function Test(mI :: Matrix{<: Color{T, 1}}) where {T}
    
    numRows, numCols = size(mI);
    dataType = T.types[1];

    mO = Matrix{dataType}(undef, numRows, numCols);
    for ii ∈ 1:numRows, jj ∈ 1:numCols
        mO[ii, jj, kk] = getfield(mI[ii, jj], 1); #<! According to ColorTypes data always in order RGBA
    end

    return mO;

end


function Test1(mI :: Matrix{<: TransparentColor{C, T, N}}) where {C, T, N}
    
    numRows, numCols = size(mI);
    numChannels = N;
    dataType = T.types[1];

    if numChannels > 1
        mO = permutedims(reinterpret(reshape, dataType, mI), (2, 3, 1));
    else
        mO = reinterpret(reshape, dataType, mI);
    end

    return mO;

end

function Test(mI :: Matrix{<: TransparentColor{C, T, N}}) where {C, T, N}
    
    numRows, numCols = size(mI);
    numChannels = N;
    dataType = T.types[1];

    if numChannels > 1
        mO = Array{dataType, 3}(undef, numRows, numCols, numChannels);
        for ii ∈ 1:numRows, jj ∈ 1:numCols
            for kk ∈ 1:numChannels
                mO[ii, jj, kk] = getfield(getfield(mI[ii, jj], kk), 1); #<! Accordign to ColorTypes data always in order RGBA
            end
        end
    else
        mO = Matrix{dataType}(undef, numRows, numCols);
        for ii ∈ 1:numRows, jj ∈ 1:numCols
            mO[ii, jj] = getfield(mI[ii, jj], 1);
        end
    end

    return mO;

end

function PrintMat( mI :: Matrix{T}, dUtfSym :: Dict ) where{T <: Integer}
    
    numRows, numCols = size(mI);

    for ii in 1:numRows
        for jj in 1:numCols
            print(dUtfSym[mI[ii, jj]]);
        end
        println();
    end

end

function FindLongestPath( mI :: Matrix{Bool} )
    # Finds the path with the highest score.
    # Uses Dynamic Programming approach.
    # The implementation is biased. Since for equality values we chose the first.
    # TODO: In case of equality change into choosing randomly.
    
    numRows, numCols = size(mI);

    mD = Matrix{Int8}(undef, numRows, numCols); #<! Directions (-1 -> Up Left, 0 -> Up, 1 -> Up Right)
    mS = Matrix{UInt8}(undef, numRows, numCols); #<! Sum of 1 in the path

    mD[1, :] .= Int8(0);
    mS[1, :] .= mI[1, :];

    for ii in 2:numRows
        for jj in 1:numCols
            jj1, jj2 = max(1, jj - 1), min(jj + 1, numCols); #<! Clipping the column indices
            rangeJ = jj1:jj2;
            maxVal, dirJ = findmax(mS[ii - 1, rangeJ]); #<! In case of equality will chose Top Left
            mS[ii, jj] = mI[ii, jj] + maxVal;
            mD[ii, jj] = rangeJ[dirJ] - jj;
        end
    end

    return mD, mS;

end

function DrawPath(mD :: Matrix{<:Signed}, mS :: Matrix{<:Unsigned}) :: Matrix{Bool}
    
    numRows, numCols = size(mD);
    
    rowIdx = size(mS, 1);
    colIdx = argmax(mS[end, :]);

    mP = zeros(Bool, numRows, numCols);

    mP[rowIdx, colIdx] = true;

    for ii in (numRows - 1):-1:1
        colIdx += mD[ii + 1, colIdx];
        colIdx = clamp(colIdx, 1, numCols);

        mP[ii, colIdx] = true;
    end

    return mP;

end


## Parameters

img001Url = "https://i.stack.imgur.com/WsSFV.png";
img002Url = "https://i.stack.imgur.com/0XAPZ.png";
img003Url = "https://i.stack.imgur.com/6aFTm.png";

## Load / Generate Data

mT = load(download(img001Url));
tSize = size(mT);

# mI = Matrix{UInt8}(undef, tSize[1], tSize[2]);

# for ii in 1:tSize[1]
#     for jj in 1:tSize[2]
#         mI[ii, jj] = mT[ii, jj].r.i;
#     end
# end

# mI = Test(mT);

# # mI = ConvertPackedImg(mT);
# mI = mI[:, :, 1]; #<! The actual image is gray

# PrintMat(mI, dUtfSymPx);

# hPlot = heatmap(mI, title = "Input Image", array = true, height = tSize[1], width = tSize[2]); #<! Display using UnicodePlots

# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end

# # Case 0
# mB = mI .== 0;
# mB = Matrix{Bool}(mB); #<! Convert to Bool

# PrintMat(mB, dUtfSymBool);

# hPlot = heatmap(mB, title = "Edges with Value 0", array = true, height = tSize[1], width = tSize[2]); #<! Display using UnicodePlots

# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end

# mD, mS = FindLongestPath(mB);

# PrintMat(mD, dUtfSymDir);

# mP = DrawPath(mD, mS);

# PrintMat(mP, dUtfSymBool);

# hPlot = heatmap(mP, title = "Path with Value 0", array = true, height = tSize[1], width = tSize[2]); #<! Display using UnicodePlots

# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end

# # Case 1
# mB = mI .== 255;
# mB = Matrix{Bool}(mB); #<! Convert to Bool

# PrintMat(mB, dUtfSymBool);

# hPlot = heatmap(mB, title = "Edges with Value 255", array = true, height = tSize[1], width = tSize[2]); #<! Display using UnicodePlots

# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end

# mD, mS = FindLongestPath(mB);

# PrintMat(mD, dUtfSymDir);

# mP = DrawPath(mD, mS);

# PrintMat(mP, dUtfSymBool);

# hPlot = heatmap(mP, title = "Path with Value 255", array = true, height = tSize[1], width = tSize[2]); #<! Display using UnicodePlots

# figureIdx += 1;
# if exportFigures
#     fileName = @sprintf "Figure%04i.png" figureIdx
#     savefig(hPlot, fileName);
# end
