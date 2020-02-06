function [ objVal ] = CalcImageObjValImgBinaryImageDistanceTransform( mA )
% ----------------------------------------------------------------------------------------------- %

sC = bwconncomp(mA);
% The maximum value of the pixels in the Distance Transform of the inverted
% image is the radius of the bounding circle and its center.
mD = bwdist(~mA);

objVal = 0;

for ii = 1:sC.NumObjects
    circleRadius = max(mD(sC.PixelIdxList{ii}));
    objVal = circleRadius * length(sC.PixelIdxList{ii});
end


end

