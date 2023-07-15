
clear();
close('all');

END_SEG_VAL = -1;

polyOrder = 1;

vY = readmatrix('vY.csv');
mW = readmatrix('mW.csv');

numSamples = size(vY, 1);

vX = linspace(0, numSamples - 1, numSamples);
vX = vX(:);

hF = figure('Position', [50, 50, 900, 400]);
hA = axes();
set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
plot(vX, vY);
title('Input Signal');
xlabel('Index');
ylabel('Value')

numSegments = size(mW, 2);

mZ = zeros(size(mW));

for ii = 1:numSegments
    lastIdx = find(mW(:, ii) == END_SEG_VAL, 1, 'first') - 1;
    vIdx = mW(1:lastIdx, ii);
    vP = polyfit(vX(vIdx), vY(vIdx), polyOrder);
    mZ(vIdx, ii) = polyval(vP, vX(vIdx));
end

vZ = sum(mZ, 2);

hF = figure('Position', [50, 50, 900, 400]);
hA = axes();
set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
set(hA, 'NextPlot', 'add');
for ii = 1:numSegments
    lastIdx = find(mW(:, ii) == END_SEG_VAL, 1, 'first') - 1;
    vIdx = mW(1:lastIdx, ii);
    plot(vX(vIdx), vY(vIdx), 'DisplayName', ['Segment ', num2str(ii, '%02d')]);
end
plot(vX, vZ, 'DisplayName', 'Piece Wise Linear Estimation');
title('Segmented Signal');
xlabel('Index');
ylabel('Value');
ClickableLegend('Location', 'southeast');

vYY = vY - vZ;

hF = figure('Position', [50, 50, 900, 400]);
hA = axes();
set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
plot(vX, vYY);
title('Residual Signal');
xlabel('Index');
ylabel('Value');
