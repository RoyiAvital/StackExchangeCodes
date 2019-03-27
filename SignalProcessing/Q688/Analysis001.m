
vI = cat(3, 255, 102, 51) / 255;
vI = cat(3, 255, 153, 51) / 255;
vI = cat(3, 255, 128, 0) / 255;
vI = cat(3, 204, 127, 102) / 255;

vI = cat(3, 255, 0, 0) / 255;
vI = cat(3, 159, 112, 96) / 255;

% Reds, Yellows, Greens, Cyans, Blues, Magentas
vPhotoshopValues = [25; 75; 35; 65; 45; 55]; %<! Photoshop Neutral

vPhotoshopValues = randi([-200, 300], [6, 1]);
% vPhotoshopValues([1, 2]) = 50;

vPhotoshopValues = [60; 50; 50; 50; 50; 50];

vCoeffValues = (vPhotoshopValues - 50) ./ 50;
mO = ApplyBlackWhiteFilter(vI, vCoeffValues);

round(mO * 255)

% vI = cat(3, 255, 102, 51) / 255;
% round(ConvertRgbToHsl(vI) * 360)
% vI = cat(3, 240, 110, 66) / 255;
% round(ConvertRgbToHsl(vI) * 360)