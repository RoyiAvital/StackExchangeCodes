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

sFiles = dir(strcat(imageSetFolderName, resizePostName, FILE_SEP, '*.jpg'));
for ii = 1:length(sFiles)
    mI = imread(strcat(sFiles(ii).folder, FILE_SEP, sFiles(ii).name));
    assert(size(mI, 1) == numRows);
    assert(size(mI, 2) == numCols);
    assert(size(mI, 3) == 3);
end

sFiles = dir(strcat(maskSetFolderName, resizePostName, FILE_SEP, '*.png'));
for ii = 1:length(sFiles)
    mI = imread(strcat(sFiles(ii).folder, FILE_SEP, sFiles(ii).name));
    assert(size(mI, 1) == numRows);
    assert(size(mI, 2) == numCols);
    assert(size(mI, 3) == 1);
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

