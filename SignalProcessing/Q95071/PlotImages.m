function [ hF, cHA, cHImgObj ] = PlotImages( tI, sPlotOpt )
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
%   -   1.0.001     17/09/2024  Royi Avital     RoyiAvital@yahoo.com
%       *   Added `supTitle`.
%   -   1.0.000     16/09/2024  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    tI (:, :, :, :) {mustBeNumeric, mustBeNonnegative, mustBeImageSet}
    sPlotOpt.zoomLevel (1, 1) {mustBePositive} = 1
    % sPlotOpt.mAlphaData (:, :) {mustBeNonnegative, mustBeReal, mustBeInRange(sPlotOpt.mAlphaData, [0, 1])} = ones(size(mI, 1), size(mI, 2));
    sPlotOpt.tAlphaData (:, :, :) {mustBeNonnegative, mustBeReal} = ones(size(tI, 2), size(tI, 3), size(tI, 1));
    sPlotOpt.showAxis (1, 1) {mustBeMember(sPlotOpt.showAxis, [0, 1])} = 0
    sPlotOpt.showTicks (1, 1) {mustBeMember(sPlotOpt.showTicks, [0, 1])} = 0
    sPlotOpt.marginSize (1, 1) {mustBeNonnegative, mustBeInteger} = 50
    sPlotOpt.titleMarginSize (1, 1) {mustBeNonnegative, mustBeInteger} = 35
    sPlotOpt.cHA (1, :) cell = {}
    sPlotOpt.cPlotTitle (1, :) cell = {}
    sPlotOpt.supTitle (1, :) cell = {}
    sPlotOpt.vSize (1, :) {mustBeInteger, mustBePositive} = [1, size(tI, 1)]
end

arguments(Output)
    hF (1, 1) {mustBeA(hF, {'matlab.ui.Figure', 'double'})}
    cHA (1, :) cell 
    cHImgObj (1, :) cell 
end

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

% Unpacking parameters struct
zoomLevel       = sPlotOpt.zoomLevel;
tAlphaData      = sPlotOpt.tAlphaData;
showAxis        = sPlotOpt.showAxis;
showTicks       = sPlotOpt.showTicks;
marginSize      = sPlotOpt.marginSize;
titleMarginSize = sPlotOpt.titleMarginSize;
cHA             = sPlotOpt.cHA;
cPlotTitle      = sPlotOpt.cPlotTitle;
supTitle        = sPlotOpt.supTitle;
vSize           = sPlotOpt.vSize;

numRows = size(tI, 2);
numCols = size(tI, 3);
numImg  = size(tI, 1);

if((~isempty(cPlotTitle)) && (length(cPlotTitle) ~= numImg))
    error('The length of "cPlotTile" must match the number of images or be an empty cell');
end

if(isempty(cPlotTitle))
    cPlotTitle = cell(1, numImg);
    for ii = 1:numImg
        cPlotTitle{ii} = '';
    end
end

numRowsEff = zoomLevel * numRows;
numColsEff = zoomLevel * numCols;

if(isempty(cHA))
    % Open a figure
    hF = figure('Units', 'pixels', 'Position', [100, 100, ((vSize(2) + 1) * marginSize) + (vSize(2) * numColsEff), ((vSize(1) + 1) * marginSize) + (vSize(1) * numRowsEff) + titleMarginSize]);
    cHA = cell(vSize(1), vSize(2));
    kk = 0;
    for ii = 1:vSize(1)
        if(kk == numImg)
            break;
        end
        for jj = 1:vSize(2)
            kk = kk + 1;
            cHA{kk} = axes(hF, 'Units', 'pixels', 'Position', [(jj * marginSize) + ((jj - 1) * numColsEff) + 1, ((vSize(1) - ii) * numRowsEff) + ((vSize(1) - ii + 1) * marginSize) + 1, numColsEff, numRowsEff]);
            if(kk == numImg)
                break;
            end
        end
    end
    cHA = cHA(1:numImg);
else
    % Axes exists
    hF = ancestor(cHA{1}, 'figure'); %<! Verifies it is an axes
end


cHImgObj = cell(1, numImg);

for ii = 1:numImg
    hA = cHA{ii};
    mI = squeeze(tI(ii, :, :, :));
    [~, ~, hImgObj] = PlotImage(mI, 'zoomLevel', zoomLevel, 'mAlphaData', tAlphaData(:, :, ii), ...
        'showAxis', showAxis, 'showTicks', showTicks, 'marginSize', ...
        marginSize, 'hA', hA, 'plotTitle', cPlotTitle{ii});
    cHImgObj{ii} = hImgObj;
end

if(~isempty(supTitle))
    sgtitle(hF, supTitle);
end


end


function [ ] = mustBeImageSet( tA )

% Test for Image Matrix

if(ndims(tA) < 3)
    error('Matrix must be IxMxN, IxMxNx1 or IxMxNx3');
end


end

