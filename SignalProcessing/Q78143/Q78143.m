% StackExchange Signal Processing Q78143
% https://dsp.stackexchange.com/questions/78143
% Reconstruction of Sparse Signal from Its Compressed Sense Spectrum
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes
% - 1.0.000     22/09/2021
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

% Signal Parameters
numSamples = 256;

% Samplling Parameters
vT = [2:2:32].';
vO = [4:4:64].';

% Experiments Parameters
numRealizations = 50;

% Reconstruction Parameters
recThr = numSamples * 1e-6;


%% Generate Data

vS = zeros(numSamples, 1);
mF = dftmtx(numSamples) / sqrt(numSamples); %<! Unitary

mFR = real(mF);
mFI = imag(mF);

mS = zeros(size(vT, 1), size(vO, 1), numRealizations);


%% Analysis

currTime = tic();

for kk = 1:numRealizations
    for jj = 1:size(vO, 1)
        numFreq = vO(jj);
        vF      = zeros(numFreq, 1);
        for ii = 1:size(vT, 1)
            numVals = vT(ii);
            
            if(numVals > numFreq)
                continue;
            end
            
            vS(:) = 0;
            
            vVIdx       = randperm(numSamples, numVals);
            vS(vVIdx)   = rand(numVals, 1);
            vFIdx       = randperm(numSamples, numFreq);
            vF(:)       = mF(vFIdx, :) * vS;
            
            % Pay attention that the solver doesn't assume Real Signal
            % (Namely symmtery could gain much more here).
            % Solvers from https://math.stackexchange.com/questions/1639716
            % The solvers doesn't support complex numbers in equality
            % constraint. Hence we break the constraint to real and
            % imaginary part. One could also build projected gradient
            % descent whcih would, probably, be faster.
            % vSRec = real(SolveBasisPursuitLp001([mFR(vFIdx, :); mFI(vFIdx, :)], [real(vF); imag(vF)]));
            vSRec = real(SolveBasisPursuitLp002([mFR(vFIdx, :); mFI(vFIdx, :)], [real(vF); imag(vF)])); %<! Much Faster
            
            mS(ii, jj, kk) = norm(vSRec - vS) <= recThr;
            
        end
    end
end

runTime = toc(currTime);

disp(['Total Run Time: ', num2str(runTime), ' [Sec]']);


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
% set(hAxes, 'Colormap', 'gray');
% colormap(hAxes, 'gray');
hImgObj = image(vO, vT, repmat(mean(mS, 3), 1, 1, 3));
colormap(hAxes, 'gray');
set(get(hAxes, 'Title'), 'String', {['Reconstruction Probability [%]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Number of Frequency Samples ($\left| \Omega \right|$)']}, ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
set(get(hAxes, 'YLabel'), 'String', {['Number of Non Zero Values ($\left| T \right|$)']}, ...
    'FontSize', fontSizeAxis, 'Interpreter', 'latex');
colorbar(hAxes);
set(hAxes, 'DataAspectRatio', [1, 1, 1]);

if(generateFigures == ON)
    % saveas(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

