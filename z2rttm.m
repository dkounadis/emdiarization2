function [] = z2rttm(filename,Z,T,E,M,fs,stft_win_len)
% z2rttm Make .rttm file from STFT frame-resolution diarization posteriors
%
% INPUTS
%
%  filename: <string>        e.g. output.rttm
%
%  Z       : [N x L]         diarization posteriors for N states, L frames
%  
%  T       : [N x N]         diarization transition matrix 
%  
%  E       : [J x N]         encoding J->N. Note that Z,T are ordered as E
%
%  M       : [1 x 1]         Number of samples in the time domain sources
%
%  fs      : [1 x 1]         sampling frequency of the time domain sources
%
% stft_win_len : [1 x 1]     length of STFT analysis window. use same as EM
%
% OUTPUTS
%
% <filename> with diarization transcription in .rttm format
%
% .rttm file format
%
% Each line in a .rttm is an interval of activity and is on the form:
% SPEAKER ID 1 0.01 2.04 <NA> <NA> source_1 <NA> 
% where 0.01 is starting time of activity in seconds
% and 2.04 is duration and source_1 is a label indicating intervals of
% the same source.

[N,L] = size(Z);

% [1 x L] integers in range 1,N
optSeq = hmmviterbi(1:L,T,Z);

% [N x L] binary selection
eD = zeros(N,L); for l=1:L, eD(optSeq(l),l) = 1; end

% [J x L] it is also binary because eD has a single ace per column 
Z = E*eD;

% [J x L] smooth
Z = ~~medfilt1(Z,50,[],2);

J = size(Z,1);

% open the .rttm file to write diarization output
fID = fopen(filename,'w');

for j=1:J
    
    % repl. frame value
    tmp = repmat(Z(j,:),stft_win_len,1);
    
    jump = stft_win_len/2;
    
    % [J x M] expand binary source signal from STFT resolution L to M time-domain samples resolution, safety append
    z = zeros(M+stft_win_len,1);
    
    % take the or of the activity of all STFT windows that overlap
    for l=0:L-1
        z(l*jump+1:l*jump+stft_win_len,:) = or( z(l*jump+1:l*jump+stft_win_len,:) , tmp(:,l+1) );
    end
    
    % look for intervals of activity, i.e. raises from 0 to 1 and vice-vers
    
    % [1 x M] fix last time-sample to 0 so to identify at least an interval
    z = z(1:M); z(1) = 0; z(end) = 0;
    
    % curr always tells a zero
    curr = 1;
    
    while curr < M
        
        % look for rise (from 0 to 1)
        startM = find( z(curr:end) == 1, 1); % it may not find a rise
        
        if isempty(startM)
            
            curr = M;
            
        else
            
            utterStart = curr + startM-1; % if empty
            
            % look for drop, skip the 0 at curr, at worst it ill say M
            durM = find( z(utterStart:end) == 0, 1);
            
            % write file
            fprintf(fID,'SPEAKER ID 1 %.2f %.2f <NA> <NA> estimatedSrc%d <NA>\n', utterStart/fs  ,  (durM-1)/fs ,  j);
            
            % curr will equal M if end is reached
            curr = utterStart+durM;
            
        end
        
    end
    
end