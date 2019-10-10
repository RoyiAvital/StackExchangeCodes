# The DFT Matrix for Non Uniform Time Samples Series

## Problem Statement

We have a signal $ x \left( t \right) $ defined on the interval $ \left[ {T}_{1}, {T}_{2} \right] $.  
Assume we have $ N $ samples of it given by $ \left\{ x \left( {t}_{i} \right) \right\}_{i = 0}^{N - 1} $. The samples time $ {t}_{i} $ is arbitrary and not necessarily uniform.

We're after the DFT of the samples $ \left\{ X \left[ k \right] \right\}_{k = 0}^{K - 1} $ as it was samples in a uniform manner (Implicitly means the samples in Frequency domain will be uniform as well).


## Deriving the Connection

In the [DFT Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) the connection between time and frequency is given by:

$$ x \left[ n \right] = \frac{1}{N} \sum_{k = 0}^{N - 1} X \left[ k \right] {e}^{j 2 \pi \frac{k}{N} n } \tag{1}  $$

In $ \eqref{EqnIdft} $ we use $ n $ for modeling the sample index in time. We usually build samples in time as $ x \left[ n \right] = x \left( n {T}_{s} \right) $ where $ {T}_{s} $ is a uniform sampling interval.  
Hence we could write:

$$ x \left( n {T}_{s} \right) = \frac{1}{N} \sum_{k = 0}^{N - 1} X \left[ k \right] {e}^{j 2 \pi \frac{k}{N {T}_{s}} n {T}_{s}} \tag{2} $$

In $ \eqref{EqnIdft2} $ we added explicit scaling of time. This is a known property of Fourier transform family which scales the domain in order to normalize the transform.

Now, there is nothing which blocks us from using arbitrary time:

$$\begin{align*} \tag{3}
x \left( t \right) & = \frac{1}{N} \sum_{k = 0}^{N - 1} X \left[ k \right] {e}^{j 2 \pi \frac{k}{N {T}_{s}} t} && \text{} \\ 
& = \frac{1}{N} \sum_{k = 0}^{N - 1} X \left[ k \right] {e}^{j 2 \pi \frac{k {F}_{s}}{N} t} && \text{Since $ {F}_{s} = \frac{1}{{T}_{s}} $}
\end{align*}$$


As can be seen $ \eqref{EqnIdft3} $ makes sense as it goes through each element according to its frequency and sums to give the output at time $ t $. We can go step farther and generalize it for cases we don't have uniform sampling frequency.  
The average sampling frequency is given by $ \bar{F}_{s} = \frac{N}{ {T}_{2} - {T}_{1} } $. Let's define $ T = {T}_{2} - {T}_{1} $ and we'll get:

$$ x \left( t \right) = \frac{1}{N} \sum_{k = 0}^{N - 1} X \left[ k \right] {e}^{ j 2 \pi k \frac{t}{T} } $$

Which is many ways resembles the [DTFT Transform](https://en.wikipedia.org/wiki/Discrete-time_Fourier_transform) equation which does the same in the other direction, transforming uniform discrete samples in time domain to arbitrary frequency (Within a frequency interval) in Frequency Domain:

$$\begin{align*} \tag{4}
X \left( f \right) & = \sum_{n = 0}^{N - 1} x \left[ n \right] {e}^{-j 2 \pi f {T}_{s} n } && \text{} \\
& = \sum_{n = 0}^{N - 1} x \left[ n \right] {e}^{-j 2 \pi \frac{f}{ {F}_{s} } n } && \text{Since $ {F}_{s} = \frac{1}{{T}_{s}} $}
\end{align*}$$

We see the same scaling, $ \frac{f}{ {F}_{s} } $ which scales the continuous $ f $ relative to the interval of frequencies $ {F}_{s} $ which is equivalent to $ \frac{t}{ T } $ which scales $ t $ relative to the time interval of the continuous signal.

## The Transform Matrix

So, given the set of time indices $ {\left\{ {t}_{i} \right\}}_{i = 0}^{N - 1} $ the transformation matrix, from frequency domain to time domain, is given by:

$$ D \in \mathbb{R}^{N \times K}, \;  {D}_{i, k} = {e}^{ j 2 \pi k \frac{ {t}_{i} }{T} } $$

## The Model

In vector form the model is:

$$ x = D y $$

Where $ y \in \mathbb{C}^{K} $ is the vector of the frequency coefficients in uniform grid, $ x $ is the samples in time (Non Uniform, Or at least no assumption of uniformity) and $ D $ as defined above.  
Since in our model we're after $ y $ the answer is given by:

$$ y = {D}^{\dagger} x $$

Where $ {D}^{\dagger} $ is the [Pseudo Inverse Matrix](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $ D $.

## Implementation & Results

The code is as following:

```matlab
subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Simulation Parameters

samplingFrequency = 101; %<! [Hz]
samplingInterval = 1 / samplingFrequency; %<! [Sec]
startTime = 1; %<! [Sec]
endTime = 4; %<! [Sec]
timeInterval = endTime - startTime; %<! [Sec]

numSamples = round(samplingFrequency * timeInterval);
numSamplesTT = round(1.2 * numSamples);

signalFreq = 2; %!< [Hz]

% The uniform time grid
vT      = linspace(startTime, endTime, numSamples + 1);
vT(end) = [];
vT      = vT(:);

% The non uniform time grid - Reconstruction
vTT = endTime * rand(numSamplesTT, 1);
vTT = sort(vTT, 'ascend');

% The non uniform time grid - DFT
vTD = linspace(startTime, endTime, (10 * numSamples) + 1);
vTD(end) = [];
vTD = vTD(sort(randperm(length(vTD), numSamples)));
vTD = vTD(:);

% The uniform frequency grid
vF      = (samplingFrequency / 2) * linspace(-1, 1, numSamples + 1);
vF(end) = [];
vF      = vF(:);

vK = [-floor(numSamples / 2):floor((numSamples - 1) / 2)];
vK = vK(:);


%% Generate Data

vX  = cos(2 * pi * signalFreq * vT);
vFx = fftshift(fft(vX));


figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = subplot(1, 2, 1);
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Reference Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);

hAxes           = subplot(1, 2, 2);
hStemObj = stem(vF, abs(vFx));
set(hStemObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['DFT of the Reference Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Frequency [Hz]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Magnitude']}, ...
    'FontSize', fontSizeTitle);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Analysis - Reconstruction

mD = exp(1j * 2 * pi * (vTT / timeInterval) * vK.') / numSamples;

% Reconstruction according to the model
vY = real(mD * vFx);

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries     = plot(vTT, vY);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end


%% Analysis - DFT of the Non Uniformly Sampled Data

vY  = cos(2 * pi * signalFreq * vTD);

mD = exp(1j * 2 * pi * (vTD / timeInterval) * vK.') / numSamples;
vFy = pinv(mD) * vY;

figureIdx = figureIdx + 1;

hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries     = plot(vT, vX);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries     = plot(vTD, vY);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Time Index']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Sample Value']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end

figureIdx = figureIdx + 1;

hFigure     = figure('Position', figPosLarge);
hAxes       = axes();
set(hAxes, 'NextPlot', 'add');
hStemObj    = stem(vF, abs([vFx, vFy]));
set(hStemObj, 'LineWidth', lineWidthNormal);
% hLineSeries     = plot(vTT, vY);
% set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Marker', '*');
set(get(hAxes, 'Title'), 'String', {['DFT of the Uniform Signal & Non Uniform Signal']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Frequency [Hz]']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'YLabel'), 'String', {['Magnitude']}, ...
    'FontSize', fontSizeTitle);
hLegend = ClickableLegend({['Uniform Signal'], ['Non Uniform Signal']});

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.png']);
end
```

Results are:

![](https://i.imgur.com/gJd1SXT.png)
![](https://i.imgur.com/Sh3ZE6x.png)
![](https://i.imgur.com/Xrg9l8E.png)
![](https://i.imgur.com/BhYwpCE.png)

## Remark: Why Do We Need to Apply `fftshift()` on the DFT of the Signal?

Indeed in the Reconstruction part we use `fftshift()`. The shallow answer is easy, we also build the vector `vK` as symmetric around zero.  
But there is a deeper reason for that. In the DFT when we use uniform sampling in Frequency Domain and Time Domain *Magic* happens without us seeing it explicitly.

When we defined the term $ \frac{k}{ N {T}_{s} } n {T}_{s} $ we replaces $ n {T}_{s} $ with $ t $ hence we prevent the term $ {T}_{s} $ to cancel itself. Now setting $ {F}_{s} = \frac{1}{{T}_{s}} $ means that we multiply by $ k $ and we get frequencies which are out of the Nyquist Frequency.  
In most cases when we that happens the Modulo property of the exponent comes in and we get the correct negative value of the frequency in the range $ \left[ -\pi, \pi \right] $. Yet when $ t $ is arbitrary we can think that $ {F}_{s} $ is changing per sample which means when we go farther than $ \pi $ the modulo doesn't bring us to the correct answer.

First, as intuition, always think the DFT is defined on the $ \left[ -\pi, \pi \right] $ interval and it is continuous. So as long as you work on this range things works as intended. This intuition can come from the Fourier Series and Discrete Fourier Series (DFS).  

Let's try explaining it using a concrete example. Let's examine the exponent term from the derivation:

$$ 2 \pi \frac{k}{N {T}_{s}} n {T}_{S} = 2 \pi \frac{k}{{F}_{S}} \frac{{F}_{s}}{N} n = 2 \pi \frac{k b}{{F}_{s}} n $$

Where $ b $ is the Bin Resolution in the Frequency domain. Now given the signal is:

$$ x \left( t \right) = \cos \left( 2 \pi f t \right) \Rightarrow x \left( n {T}_{s} \right) = \cos \left( 2 \pi f {T}_{s} n \right) \Rightarrow x \left[ n \right] = \cos \left( 2 \pi \frac{f}{ {F}_{s} } n \right) $$

For $ {F}_{s} = 100 $ [Hz] and $ N = 100 $ (Which means $ b = 1 $) we will have delta at $ k = 2 $ and $ k = 98 $. For $ k = 98 $:

$$ 2 \pi \frac{98}{{F}_{s}} n $$

This is clearly above the Nyquist frequency ($ \frac{{F}_{s}}{2} $) and only for $ {F}_{s} = 100 $ its modulo is $ -2 $ which is correct. But in the model above, since we have arbitrary $ t $ one could think we have changing $ {F}_{s} $ which means we don't get the correct value.

This means the actual equation should be:

$$ x \left( t \right) = \frac{1}{N} \sum_{k = \left \lfloor - \frac{K}{2} \right \rfloor }^{ \left \lfloor \frac{K - 1}{2} \right \rfloor } X \left[ k \right] {e}^{ j 2 \pi k \frac{t}{T} } $$