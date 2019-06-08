% Resize Images
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     07/06/2019
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

imageFolder             = 'ImageSet';
imageFileNamePattern    = '*.jpg';
vOutputSize             = [180, 120]; %<! [Rows, Columns]
imageOutputFormat       = '.jpg';


%% Resize Image

sImageData = dir(fullfile(imageFolder, imageFileNamePattern));

for ii = 1:length(sImageData)
    mI = imread(fullfile(imageFolder, sImageData(ii).name));
    mO = imresize(mI, vOutputSize);
    [~, fileName, fileExt] = fileparts(sImageData(ii).name);
    imwrite(mO, fullfile(imageFolder, [fileName, imageOutputFormat]));
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

