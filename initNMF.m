function [W,H,Kj] = initNMF(X,J,fs,micDist)
%initNMF   NMF initialization based on TDOA localization [1]
%        
% INPUTS
%
%  X       : [F x L x I]     F <freq> x L <frams> complex STFT of I sensors
%
%  J       : [1 x 1]         number of Sources to be separated (a scalar)
%
%  fs      : [1 x 1]         sampling freq. of the mix-signal in samples/s
%
% OUTPUTS
%
%  W       : [F x K]         array of bases, K = J * cPerSrc 
%
%  H       : [K x L]         array of contributions
%
%  Kj  : {J x 1} x [? x 1]   Kj is a cell-array with J elements (one for
%                            each source). Let K = size(W,2) be the total
%                            number of components. Any element Kj{j}, j=1:J
%                            of Kj contains the indexes (of the columns of
%                            W and rows of H) that correspond to source j.
%                            For example Kj = { [1 2 3] , [4 5] , [6 7] }
%                            tells that the number of sources is J = 3,
%                            and W(:,[1 2 3]) * H([1 2 3],:) is the NMF
%                            for souce j=1, W(:,[4 5]) * H([4 5],:) is
%                            the NMF of source j=2, W(:,[6 7]) * H(:,[6 7])
%                            is the NMF for source j=3.
%
%  References:
%    [1] Y. Dorfan and S. Gannot, Tree-based recursive expectation maximi-
%        zation algorithm for localization of acoustic sources, IEEE/ACM T.
%        Audio, Speech, Lang. Process., vol. 23, no. 10, pp.1692–1703, 2015
%
% version 29 September 2017, 14:03 PM

% NMF dimension for each source's sepctrum
cPerSrc = 20;

% [1 x 1] number of possible TDOA, single-iteration EM
K = 1001;   
if nargin < 5
    micDist = .2;
end
F = size(X,1);

% [K x 1] grid of TDOA, via possible angles (azimuths) in -pi/2, pi/2
tau = micDist / 340.29 * sin( linspace(-pi/2,pi/2,K) );

% [F x L] assure denominator deflation
Phase = sign(X(:,:,1))./sign( X(:,:,2) + 1e-17 );

% [F x K] candidate GMM means, 2F-1 is the win_len
mu = exp( -2j * pi * fs / (2*F-1) * (0:F-1)' * tau );

% [F x L x K] (Phase-mu), allow a singleton for L
d = bsxfun(@minus, Phase, permute(mu,[1 3 2]) );

% [F x L x K] exponents of the responsibilities, sigma = 0.3 who knows; 
d = - d.*conj(d) / .3;

% [F x L x K] responsibilities
d = exp( bsxfun(@minus, d, log(sum(exp(d),3))) );

% [1 x 1 x K] update priors exponent, because of sum on FL p may be > 0
p = log( sum(sum(d,2),1) );

% [1 x 1 x K] assure negative values
p = p - max(p,[],3);

% [1 x 1 x K] normalize
p = exp( p - log(sum(exp(p),3)) );

% [? x 1] (indices in p)
[~,ind] = findpeaks( p(:) );

% [? x 1] supplement TDOA (if found peaks are less than J)
ind = [ ind ; randi([1 K], J-numel(ind), 1) ];

% [? x 1] sort ind based on proportions p
[~,srcPk] = sort( p(ind) , 'descend' );

% [F x L x J] responsibilities of the J peaks that have the largest p
Z = d(:,:, ind( srcPk(1:J) ) );

% [F x L x J] renormalise
Z = bsxfun(@rdivide,Z,sum(Z,3));

% NMF parameters via KL and mic-1

% [F x L x J x I] Z are the binary masked source image STFTs
Z = bsxfun(@times,Z,permute(X,[1 2 4 3]));

% use 1-st mike
[W,H,Kj] = Init_KL_NMF_fr_sep_sources( Z(:,:,:,1) ,cPerSrc );
