function [ objVal ] = CalcImageObjValImgBinaryImageProps( mA )
% ----------------------------------------------------------------------------------------------- %

sImgProps = regionprops(mA, {'Area', 'MajorAxisLength'});

objVal = 0;

for ii = 1:length(sImgProps)
    majorAxisLength = sImgProps(ii).MajorAxisLength;
    circleRadius = majorAxisLength / 2;
    objVal = circleRadius * sImgProps(ii).Area;
end


end

