
rng(23);

clear();
close('all');

numSamples    = 500;
ampFiltSize   = 20;
phaseFiltSize = 50;

vAmp    = rand(numSamples, 1);
vAmp    = 0.2 * conv2(vAmp, ones(ampFiltSize, 1) / ampFiltSize, 'same');
vPhase  = 0.2 * rand(numSamples, 1);
vPhase  = conv2(vPhase, ones(phaseFiltSize, 1) / phaseFiltSize, 'same');
vPhase  = cumsum(vPhase);

vX = linspace(0, numSamples - 1, numSamples);
vX = vX(:);

vC = vAmp .* cos(2 * pi * vPhase);
vL = zeros(numSamples, 1);
vL(001:150) = 0;
vL(151:300) = 1;
vL(301:400) = linspace(0.5, 1.0, 100);
vL(401:500) = linspace(1.0, 0.4, 100);

vY = vC + vL;

hF = figure('Position', [50, 50, 900, 400]);
hA = axes();
set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);
plot(vX, vY);
title('Input Signal');
xlabel('Index');
ylabel('Value');

writematrix(vY, 'vY.csv');

