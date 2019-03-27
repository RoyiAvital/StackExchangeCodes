
vI = cat(3, 85, 80, 217) / 255;

vPhotoshopValues = [30; 98; 51; 73; 5; 53];

vCoeffValues = (vPhotoshopValues - 50) ./ 50;
vO = ApplyBlackWhiteFilter(vI, vCoeffValues);
vO = max(min(vO, 1), 0);

round(255 * vI(:))
round(255 * vO(:))