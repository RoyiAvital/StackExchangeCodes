function [ uNet ] = BuildUNet( numBlocks, numFiltersBase, dropOutProb, batchNormFlag, numChannelsOutput )
% ----------------------------------------------------------------------------------------------- %
%[ estFreq ] = EstimateSineFreqKay( vX, samplingFreq, estType )
% Estimates the frequency of a single Real Harmonic signal with arbitrary
% phase.
% Input:
%   - vX                -   Input Samples.
%                           The vector to be optimized. Initialization of
%                           the iterative process.
%                           Structure: Vector (numSamples X 1).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - paramLambda       -   Parameter Lambda.
%                           The L1 Regularization parameter.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - numIterations     -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% Output:
%   - estFreq           -   Number of Iterations.
%                           Number of iterations of the algorithm.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range {1, 2, ...}.
% References
%   1.  Steven Kay - A Fast and Accurate Single Frequency Estimator.
% Remarks:
%   1.  It would work with complex numbers (Harmonic Signla) by changing:
%       vX(ii) * vX(ii + 1) -> vX(ii)' * vX(ii + 1).
%   2.  fds
% Known Issues:
%   1.  C
% TODO:
%   1.  D
% Release Notes:
%   -   1.0.000     09/08/2021  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    numBlocks (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBePositive}
    numFiltersBase (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBePositive}
    dropOutProb (1, 1) {mustBeNumeric, mustBeReal, mustBeNonnegative} = 0.1
    batchNormFlag (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(batchNormFlag, [0, 1])} = 1
    numChannelsOutput (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger} = 3
end

kernelSize = 3;

uNet       = layerGraph();
numFilters = numFiltersBase;

% Encoder
for ii = 1:numBlocks
    uNet = addLayers(uNet, UNetConv2dBlock(numFilters, kernelSize, batchNormFlag, ['EncoderBlock', num2str(ii, '%03d')]));
    uNet = addLayers(uNet, maxPooling2dLayer([2, 2], 'Name', ['EncoderMaxPoolLayer', num2str(ii, '%03d')], 'Stride', 2));
    uNet = addLayers(uNet, dropoutLayer(dropOutProb, 'Name', ['EncoderDropOutLayer', num2str(ii, '%03d')]));
    numFilters = numFilters * 2;
end

numFilters = numFilters * 2;
uNet = addLayers(uNet, UNetConv2dBlock(numFilters, kernelSize, batchNormFlag, ['EncoderBlock', num2str(ii + 1, '%03d')]));

% Decoder
for ii = 1:numBlocks
    numFilters = numFilters / 2;
    uNet = addLayers(uNet, transposedConv2dLayer(3, numFilters, 'Stride', 2, 'Cropping', 'same', 'Name', ['DecoderConv2DT', num2str(ii, '%03d')]));
    uNet = addLayers(uNet, depthConcatenationLayer(2, 'Name', ['DecoderConCatLayer', num2str(ii, '%03d')]));
    uNet = addLayers(uNet, dropoutLayer(dropOutProb, 'Name', ['DecoderDropOutLayer', num2str(ii, '%03d')]));
    uNet = addLayers(uNet, UNetConv2dBlock(numFilters, kernelSize, batchNormFlag, ['DecoderBlock', num2str(ii, '%03d')]));
end

% Output layer
uNet = addLayers(uNet, convolution2dLayer(1, numChannelsOutput, 'Name', 'OutputConv'));

% Trick to make all layers sequential
uNet = layerGraph(uNet.Layers);

% Connecting the skip layers
jj = numBlocks; %<! Decoder
for ii = 1:numBlocks
    uNet = uNet.connectLayers(['EncoderBlock', num2str(ii, '%03d'), 'ReluLayer2'], ['DecoderConCatLayer', num2str(jj, '%03d'), '/in2']);
    jj = jj - 1;
end


end


function [ blockLayers ] = UNetConv2dBlock( numFilters, kernelSize, batchNormFlag, baseName )


arguments
    numFilters (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBePositive}
    kernelSize (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeNonnegative}
    batchNormFlag (1, 1) {mustBeNumeric, mustBeReal, mustBeInteger, mustBeMember(batchNormFlag, [0, 1])}
    baseName (1, :) {mustBeTextScalar} = '';
end

% blockLayers = layerGraph();
% 
% % First layer
% blockLayers = addLayers(blockLayers, convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', 'BlockConv2DLayer1'));
% if(batchNormFlag)
%     blockLayers = addLayers(blockLayers, batchNormalizationLayer('Name', [baseName, 'BlockBatchNormLayer1']));
% end
% blockLayers = addLayers(blockLayers, reluLayer('Name', [baseName, 'BlockReluLayer1']));
% 
% % Second layer
% blockLayers = addLayers(blockLayers, convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', [baseName, 'BlockConv2DLayer2']));
% if(batchNormFlag)
%     blockLayers = addLayers(blockLayers, batchNormalizationLayer('Name', [baseName, 'BlockBatchNormLayer2']));
% end
% blockLayers = addLayers(blockLayers, reluLayer('Name', [baseName, 'BlockReluLayer2']));

if(batchNormFlag)
    blockLayers = [
        convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', [baseName, 'Conv2DLayer1'])
        batchNormalizationLayer('Name', [baseName, 'BatchNormLayer1'])
        reluLayer('Name', [baseName, 'BlockReluLayer1'])
        convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', [baseName, 'Conv2DLayer2'])
        batchNormalizationLayer('Name', [baseName, 'BatchNormLayer2'])
        reluLayer('Name', [baseName, 'ReluLayer2'])];
else
    blockLayers = [
        convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', [baseName, 'Conv2DLayer1'])
        reluLayer('Name', [baseName, 'ReluLayer1'])
        convolution2dLayer(kernelSize, numFilters, 'Padding', 'same', 'Name', [baseName, 'Conv2DLayer2'])
        reluLayer('Name', [baseName, 'ReluLayer2'])];
end


end

