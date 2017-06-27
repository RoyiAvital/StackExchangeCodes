********** Files %%%%%%%%%%%%%%%
handel.wav   %true audio signal
input.wav      %input audio signal
handel.mat  % MATLAB data file of handel which can be loaded via calling "load handel"
input.mat     % MATLAB data file of handel which can be loaded via calling "load input"

kernel h is as follows
h=[0.0545    0.2442    0.4026    0.2442    0.0545]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% How to read, save, and play audio files %%%%%%%
Depending on the version of MATLAB, 
either using the following two files: wavread, wavwrite,
or using  the following two files: audioread, audiowrite.

%%%% Sample code in MATLAB %%%%%%%%%%%%
[handel,FS]=audioread('handel.wav');  % reading wav file
audiowrite('handel.wav',handel,FS);      % writing wav file
The wav file can be played either in MATLAB via calling "audioplayer"
or played by some software.
