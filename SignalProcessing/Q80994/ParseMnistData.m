function [ tDigitImg, vDigitLabel ] = ParseMnistData( imgFile, labelFile, trimImg, scaleImg )
% ----------------------------------------------------------------------------------------------- %
%[tDigitImg, vDigitLabel] = ParseMnistData( imgFile, labelFile )
% Parses MNIST data set into a tensor of images and a vector of labels.
% Input:
%   - imgFile           -   Full Path to Images File.
%                           Full path to images file in MNIST format.
%                           Structure: Vector.
%                           Type: 'Char'.
%                           Range: NA.
%   - labelFile         -   Full Path to Labels File.
%                           Full path to labels file in MNIST format.
%                           Structure: Vector.
%                           Type: 'Char'.
%                           Range: NA.
%   - trimImg           -   Trim Image.
%                           If set to 1 will trim images into 20x20
%                           (Center).
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - scaleImg          -   Scale Image.
%                           If set to 1 will trim images into the [0, 1]
%                           range.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
% Output:
%   - tDigitImg         -   Digit Images Tensor.
%                           A tensor where each 3rd dimension slice is a
%                           digit image.
%                           Structure: Tensor (numRowx x numCols x numImages).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 255].
%   - vDigitLabel       -   Image Labels Vector.
%                           A vector where its 'ii' element is the label of
%                           the 'ii' image in the images tensor.
%                           Structure: Vector (numImages X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 9].
% References
%   1.  A
% Remarks:
%   1.  It works on 2 files: Images, Labels.
% Known Issues:
%   1.  C
% TODO:
%   1.  D
% Release Notes:
%   -   1.0.000     30/09/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    imgFile char {mustBeFile, mustBeVector}
    labelFile char {mustBeFile, mustBeVector}
    trimImg (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(trimImg, [0, 1])} = 0
    scaleImg (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(scaleImg, [0, 1])} = 1
end

IMG_FILE_HEADER     = 2051;
LABEL_FILE_HEADER   = 2049;
    
% Parse the digit images
fileID = fopen(imgFile, 'r', 'b');
fileHeader = fread(fileID, 1, 'int32');
if(fileHeader ~= IMG_FILE_HEADER)
    error('Invalid digits images file header');
end

numImages = fread(fileID, 1, 'int32');

numRows = fread(fileID, 1, 'int32');
numCols = fread(fileID, 1, 'int32');

tDigitImg = zeros(numRows, numCols, numImages);

for kk = 1:numImages
    for ii = 1:numRows
        tDigitImg(ii, :, kk) = fread(fileID, numCols, 'uint8');
    end
end

% Close file, Finished parsing images
fclose(fileID);

% Read digit labels
fileID = fopen(labelFile, 'r', 'b');
fileHeader = fread(fileID, 1, 'int32');
if(fileHeader ~= LABEL_FILE_HEADER)
    error('Invalid label file header');
end
numLabels = fread(fileID, 1, 'int32');

if(numLabels ~= numImages)
    error('The number of images doesn''t match the number of lables')
end

vDigitLabel = fread(fileID, numLabels, 'uint8');
% Close file, Finished parsing labels
fclose(fileID);

if(trimImg)
    % Returns the center 20x20 image
    tDigitImg = tDigitImg(5:24, 5:24, :);
end

if(scaleImg)
    tDigitImg = tDigitImg ./ 255;
end

    
end
