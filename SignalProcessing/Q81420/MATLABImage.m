
clear();
close('all');

mI = imread('https://i.imgur.com/gx7rrPS.png');
numRows = size(mI, 1);
numCols = size(mI, 2);

marginSize = 50;

hF = figure('Position', [50, 50, numCols + 2 * marginSize, numRows + 2 * marginSize]); %<! Position in pixels
%  Creating axes with the size of the image (Inner space) Margins of `marginSize` pixels in each direction
hA = axes(hF, 'Units', 'pixels', 'Position', [marginSize, marginSize, numCols, numRows]); %<! Drawing axes inside the figure
image(hA, repmat(mI, 1, 1, 3)); %<! Displayed within the axes
% If removing all bounding lines is needed
% set(hA, 'XTick', [], 'XTickLabel', []);
% set(hA, 'YTick', [], 'YTickLabel', []);
% set(hA, 'Box', 'off');
% set(hA, 'XGrid', 'off', 'YGrid', 'off');
set(get(hA, 'XAxis'), 'Visible', 'off');
set(get(hA, 'YAxis'), 'Visible', 'off');

print(hF, ['Figure.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution