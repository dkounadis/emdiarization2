clear
close all
addpath aux_tools/

% length of the STFT analysis window
stft_win_len = 512;

% number of EM iterations
maxNumIter = 10;

% number of sources to be estimated from the EM
J = 2;

% load mix signal from .wav file
[x,fs] = audioread('mixture.wav');    x = transpose(x);

% call dnd.m to separate and diarise x
ye = dnd(x,J,fs,maxNumIter,stft_win_len);

% .wav's of separated source images
arrayfun(@(j) audiowrite(sprintf('estimatedSrc%d.wav',j), ye(:,:,j) ,fs), 1:J , 'uniformoutput',false)




