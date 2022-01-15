% StackExchange Signal Processing Q80994
% https://dsp.stackexchange.com/questions/80994
% Building a Pipeline for Image Classification / Clustering Tasks with Features Extractor (Example on MNIST Data)
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     29/12/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

dataSetBaseUrl      = 'http://yann.lecun.com/exdb/mnist/';
imageSetUrl         = 'train-images-idx3-ubyte.gz'; %<! http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
annotationSetUrl    = 'train-labels-idx1-ubyte.gz'; %<! http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

imageSetFileName        = 'ImageSet';
annotationSetFileName   = 'AnnotationSet';

fileExt = '.gz';

% Parsing Parameters
trimImg     = 1;
sclaeImg    = 1;

% Train / Test Partition
numTestImg = 5000;


%% Generate / Load Data

if(~exist(strcat(imageSetFileName, fileExt), 'file'))
    websave(strcat(imageSetFileName, fileExt), strcat(dataSetBaseUrl, imageSetUrl));
end
if(~exist(strcat(annotationSetFileName, fileExt), 'file'))
    websave(strcat(annotationSetFileName, fileExt), strcat(dataSetBaseUrl, annotationSetUrl));
end

if(~isfile(imageSetFileName))
    gunzip(strcat(imageSetFileName, fileExt));
end
if(~isfile(annotationSetFileName))
    gunzip(strcat(annotationSetFileName, fileExt));
end

[tImg, vLabel] = ParseMnistData(imageSetFileName, annotationSetFileName, trimImg, sclaeImg);

numRows = size(tImg, 1);
numCols = size(tImg, 2);
numImg = size(tImg, 3);


%% Train / Test Partition

vTestSet    = randperm(numImg, numTestImg);
vTrainSet   = 1:numImg;
vTrainSet(vTestSet) = [];

tTrainSetImg    = tImg(:, :, vTrainSet);
vTrainSetLabel  = vLabel(vTrainSet);
tTestSetImg     = tImg(:, :, vTestSet);
vTestSetLabel   = vLabel(vTestSet);

save('MNISTDataSet', 'tTrainSetImg', 'vTrainSetLabel', 'tTestSetImg', 'vTestSetLabel');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

