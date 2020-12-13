clear; close all;

%% 10MHz incident signal;
fs = 1000e6;
f = 10e6;
wavelength = fs/f;
sig = wavemaker(3.5, f, fs);
figure; subplot(3,1,1);
plot(sig, 'LineWidth', 1); title(strcat('input signal, wavelength =',num2str(wavelength),' data pts'));
axis([0,3300,-2,2])

%% Sparse signal;
N = 3000;                    % N : length of signal
s = zeros(N,1);
k = [50:(50+wavelength*0.1) 500:(500+wavelength*0.6) 1200:(1200+wavelength*1.6) 2200:(2200+wavelength*4.6)];
s(k) = 1;
subplot(3,1,2);
plot(s, 'LineWidth', 1); title('distribution: objective function');
axis([0,3300,0,2])


%% convoluted signal
y = conv(sig,s);
subplot(3,1,3);plot(y, 'LineWidth', 1);hold on;
plot(abs(hilbert(y)), 'LineWidth', 1);
title('convoluted signal between input and districution');
xlim([0,3300])

% My addition
vX = s(:);
vH = sig(:);
vY = y(:);

save('UserData', 'vX', 'vY', 'vH');
% End of my addition


function x = wavemaker(nCycles, fc, fs)
% function to generate wave packet;
nSample = round(fs / fc * nCycles);
ts      = 1 / fs;
T       = ts * nSample;
t_max   = ts * (nSample-1);
t       = 0: ts: t_max;
 
x       = sin( 2 * pi * fc .* t);
  
x = x.*hanning(nSample)';
end