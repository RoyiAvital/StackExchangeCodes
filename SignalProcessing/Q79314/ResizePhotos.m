% StackExchange Signal Processing Q79314
% https://dsp.stackexchange.com/questions/79314
% Image Segmentation Using Deep Learning
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     27/11/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageSetFolderName  = 'ImageSet';
maskSetFolderName   = 'MaskSet';

resizePostName = 'Resize';

numRows     = 64; 
numCols     = 64; 


%% Generate Data

if(exist(strcat(imageSetFolderName, resizePostName), 'dir'))
    rmdir(strcat(imageSetFolderName, resizePostName), 's');
end
if(exist(strcat(maskSetFolderName, resizePostName), 'dir'))
    rmdir(strcat(maskSetFolderName, resizePostName), 's');
end

mkdir(strcat(imageSetFolderName, resizePostName));
mkdir(strcat(maskSetFolderName, resizePostName));

sFiles = dir(strcat(imageSetFolderName, FILE_SEP, '*.jpg'));
for ii = 1:length(sFiles)
    mI = imread(strcat(imageSetFolderName, FILE_SEP, sFiles(ii).name));
    mI = imresize(mI, [numRows, numCols]);
    if(size(mI, 3) == 1)
        mI = repmat(mI, [1, 1, 3]);
    end
    imwrite(mI, strcat(imageSetFolderName, resizePostName, FILE_SEP, sFiles(ii).name))
end

sFiles = dir(strcat(maskSetFolderName, FILE_SEP, '*.png'));
for ii = 1:length(sFiles)
    mI = imread(strcat(maskSetFolderName, FILE_SEP, sFiles(ii).name));
    mI = imresize(mI, [numRows, numCols]);
    imwrite(mI, strcat(maskSetFolderName, resizePostName, FILE_SEP, sFiles(ii).name))
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

