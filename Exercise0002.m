% Computational Methods - Exercise 002
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     31/05/2017  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Question 0003 Part I

% Part A
numRows = 7;
vC = randi([-10, 10], [numRows, 1]);

cvx_begin quiet
    variable vX(numRows)
    minimize( vC.' * vX )
    subject to 
        sum_square(vX) <= 9
cvx_end

vX.' * vX;

vXX = (-3 * vC) / norm(vC);

[vX, vXX]

% Part B

vC = [2; 0];

vL = [0; 3];
vU = [2; 4];

vX = linprog(vC, [], [], [], [], vL, vU);



%% Question 0003 Part II

numPts  = 1000;
vX      = linspace(-1, 1, numPts);
mZ      = nan([numPts, numPts]);

mA = [1 , 1; 1 -1; -1, 1; -1, -1];
vB = [1; 1; 1; 1];

for ii = 1:numPts
    for jj = 1:numPts
        vV = [vX(ii); vX(jj)];
        if(all(mA * vV <= vB))
            mZ(ii, jj) = max(vV) - min(vV);
            % mZ(ii, jj) = 1;
        end
    end
end


figureIdx       = figureIdx + 1;
hFigure         = figure('Position', [100, 100, 1300, 850]);
hAxes           = axes();
hChartSurface   = mesh(vX, vX, mZ);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(get(hAxes, 'Title'), 'String', {['']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['y']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'ZLabel'), 'String', {['f(x, y)']}, ...
    'FontSize', fontSizeAxis);


%% Question 0003 Part III
% See Problem 5.5 In Boyd Additional Exercise (Search for "Regression")

numRows = 20;
numCols = 7;

mA = randn([numRows, numCols]);
vB = randn([numRows, 1]);

vXL2 = mA \ vB;

cvx_begin quiet
    variable vXL15(numCols)
    minimize( norm( ((mA * vXL15) - vB), 1.5 ) )
cvx_end

% Optimaliity Condition
vObjGrad = 0;
for ii = 1:numRows
    currVal = (mA(ii, :) * vXL15) - vB(ii);
    % currCons = sqrt(abs(currVal)) * sign(currVal) * mA(ii, :).';
    vObjGrad = vObjGrad + (sqrt(abs(currVal)) * sign(currVal) * mA(ii, :).');
end

cvx_begin quiet
    variables vT(numRows) vS(numRows) vXX(numCols)
    minimize( sum(vT) )
    subject to 
        vS .^ 1.5 <= vT
        vS >= 0
        mA * vXX - vB <= vS
        mA * vXX - vB >= -vS
cvx_end

cvx_begin quiet
    variables vT(numRows) vS(numRows) vS1(numRows) vS2(numRows) vY(numRows) vXXX(numCols)
    minimize( sum(vT) )
    subject to 
        for ii = 1:numRows
            [vS(ii), vY(ii); vY(ii), 1] == semidefinite(2);
            [vY(ii), vS(ii); vS(ii), vT(ii)] == semidefinite(2);
        end
        mA * vXXX + vS - vS1 == vB;
        mA * vXXX - vS + vS2 == vB;
        vS1 >= 0;
        vS2 >= 0;
cvx_end

[vXL2, vXL15, vXX, vXXX]


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

