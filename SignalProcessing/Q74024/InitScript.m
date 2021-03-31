% Init Script
% Remarks:
%   1.  If there is a predefined variable called 'subStreamNumberDefault'
%       it will be used to set the SubStream of the Random Number
%       Generator.
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.004     05/07/2018  Royi Avital
%   *   Added case for 'subStreamNumberDefault = 0'.
% - 1.0.003     29/06/2018  Royi Avital
%   *   Clearing all variables but 'subStreamNumberDefault'.
%   *   Setting 'subStreamNumber' only if 'subStreamNumberDefault' not
%       defined.
% - 1.0.002     19/04/2018  Royi Avital
%   *   Added 'markerSizeMedium' (Matches 'markerSizeNormal').
% - 1.0.001     24/11/2016  Royi Avital
%   *   Added figure positions options.
% - 1.0.000     08/11/2016  Royi Avital
%   *   First release.
%
%
%% General Parameters

close('all');
clearvars('-except', 'subStreamNumberDefault');
clc();

% set(0, 'DefaultFigureWindowStyle', 'docked');
% defaultLoosInset = get(0, 'DefaultAxesLooseInset');
% set(0, 'DefaultAxesLooseInset', [0.05, 0.05, 0.05, 0.05]);

figPosSmall     = [100, 100, 0400, 0300];
figPosMedium    = [100, 100, 0800, 0600];
figPosLarge     = [100, 100, 0960, 0720];
figPosXLarge    = [100, 100, 1100, 0825];
figPosX2Large   = [100, 100, 1200, 0900];
figPosX3Large   = [100, 100, 1400, 1050];
figPosDefault   = [100, 100, 0560, 0420];

fontSizeTitle   = 14;
fontSizeAxis    = 12;
fontSizeString  = 12;

lineWidthThin   = 1;
lineWidthNormal = 3;
lineWidthThick  = 4;

markerSizeSmall     = 4;
markerSizeMedium    = 8;
markerSizeNormal    = 8;
markerSizeLarge     = 10;

% https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html
% https://www.mathworks.com/matlabcentral/answers/160332
mColorOrder = get(groot, 'DefaultAxesColorOrder');

randomNumberStream  = RandStream('mlfg6331_64', 'NormalTransform', 'Ziggurat');

if(exist('subStreamNumberDefault', 'var') && (subStreamNumberDefault ~= 0))
    subStreamNumber = subStreamNumberDefault;
else
    subStreamNumber = round(sum(clock()));
end
set(randomNumberStream, 'Substream', subStreamNumber);
RandStream.setGlobalStream(randomNumberStream);

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

