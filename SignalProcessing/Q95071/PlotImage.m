function [ hF, hA, hImgObj ] = PlotImage( mI, sPlotOpt )
% ----------------------------------------------------------------------------------------------- %
% [ hF, hA, hL ] = PlotDft( mX, samplingFrequency, sPlotDftOpt )
%   Plots the DFT of the input signal (Or matrix of signals as columns).
% Input:
%   - mI                -   Input Image.
%                           Structure: Matrix (numRows x numCols x numChannels).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - zoomLevel         -   Zoom Level.
%                           The zoom level of the image.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - mAlphaData        -   The Alpha Channel of the Image.
%                           Structure: Matrix (numRows x numCols).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
%   - showAxis          -   Show Axis.
%                           If set to 1 axis will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - showTicks         -   Show Axis Ticks.
%                           If set to 1 the axis ticks will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - marginSize        -   Margin Size.
%                           The margin around the axes in pixels.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1, 2, ...}.
%   - hA                -   Axes Handler.
%                           The axes handler to use for the plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - openFig           -   Open a Figure.
%                           If set to 1 a new figure will be used.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - plotTitle         -   The Plot Title String.
%                           The string to plot as title. If empty, no title
%                           will be displayed.
%                           Structure: String.
%                           Type: String.
%                           Range: NA.
% Output:
%   - hF                -   Figure Handler.
%                           The figure handler of the output plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - hA                -   Axes Handler.
%                           The axes handler of the output plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - hImgObj           -   Image Object Handler.
%                           The image object handler.
%                           Structure: Scalar.
%                           Type: Handler / Object.
%                           Range: NA.
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
%   Release Notes:
%   -   1.1.000     24/11/2023  Royi Avital     RoyiAvital@yahoo.com
%       *   Removed the `openFig` option. Set by `hA`.
%   -   1.0.000     24/11/2023  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mI (:, :, :) {mustBeNumeric, mustBeNonnegative, mustBeImageMatrix}
    sPlotOpt.zoomLevel (1, 1) {mustBePositive} = 1
    % sPlotOpt.mAlphaData (:, :) {mustBeNonnegative, mustBeReal, mustBeInRange(sPlotOpt.mAlphaData, [0, 1])} = ones(size(mI, 1), size(mI, 2));
    sPlotOpt.mAlphaData (:, :) {mustBeNonnegative, mustBeReal} = ones(size(mI, 1), size(mI, 2));
    sPlotOpt.showAxis (1, 1) {mustBeMember(sPlotOpt.showAxis, [0, 1])} = 0
    sPlotOpt.showTicks (1, 1) {mustBeMember(sPlotOpt.showTicks, [0, 1])} = 0
    sPlotOpt.marginSize (1, 1) {mustBeNonnegative, mustBeInteger} = 50
    sPlotOpt.hA (1, 1) {mustBeA(sPlotOpt.hA, ["matlab.graphics.axis.Axes", 'double'])} = 0
    sPlotOpt.plotTitle (1, :) string = {['']}
end

arguments(Output)
    hF (1, 1) {mustBeA(hF, {'matlab.ui.Figure', 'double'})}
    hA (1, 1) {mustBeA(hA, {'matlab.graphics.axis.Axes', 'double'})}
    hImgObj (1, 1) {mustBeA(hImgObj, {'matlab.graphics.primitive.Image', 'double'})}
end

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

% Unpacking parameters struct
zoomLevel   = sPlotOpt.zoomLevel;
mAlphaData  = sPlotOpt.mAlphaData;
showAxis    = sPlotOpt.showAxis;
showTicks   = sPlotOpt.showTicks;
marginSize  = sPlotOpt.marginSize;
hA          = sPlotOpt.hA;
plotTitle   = sPlotOpt.plotTitle;

numRows = size(mI, 1);
numCols = size(mI, 2);

numRowsEff = zoomLevel * numRows;
numColsEff = zoomLevel * numCols;

if(hA == 0)
    hF = figure('Position', [100, 100, 2 * marginSize + numColsEff, 2 * marginSize + numRowsEff]);
    hA = axes(hF, 'Units', 'pixels', 'Position', [marginSize + 1, marginSize + 1, numColsEff, numRowsEff]);
else
    % Axes handler exists
    hF = ancestor(hA, 'figure');
end

numChannels = size(mI, 3);
if(numChannels == 1)
    mI = repmat(mI, 1, 1, 3);
end

hImgObj = image(hA, mI, 'AlphaData', mAlphaData);
if(~isempty(plotTitle))
    set(get(hA, 'Title'), 'String', plotTitle, 'FontSize', 14);
end


end


function [ ] = mustBeImageMatrix( mA )

% Test for Image Matrix

if((size(mA, 3) ~= 1) && (size(mA, 3) ~= 3))
    error('Matrix must be MxN, MxNx1 or MxNx3');
end


end

