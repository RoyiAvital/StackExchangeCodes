A = readmatrix('mA.csv');
R = readmatrix('mR.csv');
Y = readmatrix('mY.csv');

% Ynoisy = awgn(Y, 10, 'measured'); % Add noise with 10dB signal-to-noise ratio.
tic()
Yindependent = Y * inv(A);   % Remove crosstalk between channels.

X = zeros(size(Y));
for channel = 1:size(Y,2)         % Deconvolve each channel individually with NNLS.
    X(:,channel) = lsqnonneg(R, Yindependent(:,channel));
end
toc()