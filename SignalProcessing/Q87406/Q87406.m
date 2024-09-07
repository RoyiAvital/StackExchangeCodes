% StackExchange Signal Processing Q87406
% https://dsp.stackexchange.com/questions/87406
% The Different Solution for Filter Coefficients for Periodic Convolution and Full Convolution
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     06/04/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants


%% Parameters

% Short
numSamples  = 7; %<! For x
numCoeff    = 5; %<! for h

% Long
% numSamples  = 17; %<! For x
% numCoeff    = 5; %<! for h


%% Generate / Load Data

vX = reshape(1:numSamples, numSamples, 1);
vH = 1:numCoeff;
vH = vH(:) / sum(vH);

vYFull      = conv(vX, vH, 'full');
vYCyclic    = cconv(vX, vH, numSamples); %<! Matches ifft(fft(vX) .* fft(vH, numSamples), 'symmetric')

mXFull      = BuildConvFullMat(vX, numCoeff);
mXCyclic    = BuildConvCyclicMat(vX, numCoeff);

max(abs(mXFull * vH - vYFull))
max(abs(mXCyclic * vH - vYCyclic))


%% Analysis

hF = figure('Position', [100, 100, 800, 800]);

hA = subplot(1, 2, 1);
PlotMatrix(mXFull, 'titleString', 'Full Convolution');

hA = subplot(1, 2, 2);
PlotMatrix(mXCyclic, 'titleString', 'Cyclic Convolution');


%% Display Results


disp(['The reference `vH`:' , num2str(vH.')]);
disp(['The reference `vH`:' , num2str(vH.')]);

% figureIdx = figureIdx + 1;
% 
% hF = figure('Position', figPosLarge);
% hA   = axes(hF);
% set(hA, 'NextPlot', 'add');
% hLineObj = plot(-mU1(:, 1), 'DisplayName', 'Kernel A');
% set(hLineObj, 'LineWidth', lineWidthNormal);
% hLineObj = plot(-mU2(:, 1), 'DisplayName', 'Kernel B');
% set(hLineObj, 'LineWidth', lineWidthNormal);
% set(get(hA, 'Title'), 'String', {['Separable Filters of the Kernels']}, ...
%     'FontSize', fontSizeTitle);
% hLegend = ClickableLegend();
% 
% if(generateFigures == ON)
%     % saveas(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
%     print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
% end



%% Auxiliary Functions

function [ mX ] = BuildConvFullMat( vK, numSamples )

vC = [vK(:); zeros(numSamples - 1, 1)];
vR = [vK(1); zeros(numSamples - 1, 1)];

mX = toeplitz(vC, vR);

end

function [ mX ] = BuildConvCyclicMat( vK, numSamples )

vC = vK(:);

% Works for length(vK) >= numSamples + 1
% vR = [vK(1); flip(vK((end - numSamples + 2):end))];

% Works for any relation between numSamples and length(vK)
vR = padarray(vK(:), numSamples, 'circular', 'pre');
vR = flip(vR(2:(numSamples + 1)));

mX = toeplitz(vC, vR);

end




%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

