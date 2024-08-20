function y = dnd(x,J,fs,maxNumIter,stft_win_len)
%DND  Duong N Diarisation function that implements [1]
%
% INPUTS
%
%  x          : [I x M]      time domain mixture with M samples and I mikes
%
%  J          : scalar       number of sources
%
%  fs         : scalar       sampling frequency of x, e.g. 16000
%
%  maxNumIter : scalar       number of EM iterations
%
%  stft_win_len : scalar     STFT analysis window (in samples) e.g. 512
%
% OUTPUT
%
%  y  : [M x I x J]          estimates of the J source images (time domain)
%
%
%  References:
%    [1] D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, R. Horaud, 
%        S. Gannot. Exploiting The Intermittency of Speech for Joint
%        Separation and Diarisation, in Proc. of WASPAA 2017.
%
% v. September 29 2017, 14:04 PM
fprintf('[Duong n Diarization v1.0] September 29, 2017, 14:04 PM\n');




%    /\
%   /__\
%  /    \ R C H E T Y P E



%% A   Constants indexSets & functions

% [F x L x I] STFT of input mixture
X = stft_multi( x, stft_win_len);

% sizes
[F,L,I] = size(X);   N = pow2(J); M = size(x,2);

% initialise NMF matrices with [2]
[W,H,Kj] = initNMF(X,J,fs,2);

% [J x N] states of diarisation, alternat. de2bi(0:N-1,J,'left-msb')'
E = floor( mod( bsxfun(@rdivide, 0 : N-1 , pow2(J-1:-1:0)' ) , 2 ) );

% [1 x 1 x N x J] all mtx to F x L x N x J x I x I
E = permute(E,[3 4 2 1]);

% [F x L] norms of X
normX = sum( sum( X .* conj(X) , 2 ) , 3 );

% f(x) normalize a matrix by the sum of its columns
normalize = @(Z) bsxfun(@rdivide,Z,sum(Z,1));

% [N x N] T, [1 x 1 x N] l=1 prior
T = rand(N);    T = T/sum(T(:));    Z = ones(1,1,N)/N;

% [F x 1] sensor
v = X(:)' * X(:) / numel(X);

% [1 x 1 x 1 x 1 x I x I] offset
eyeI = zeros(F,1,1,J,I,I);    eyeI(:,:,:,:,1:I+1:I*I) = 1;    R = eyeI;

% [F x L x 1 x 1 x I] indexing dimensions
X = permute(X,[1 2 4 5 3]);

% f(x) matrix-vector product, the matrix is in dim 5x6, vector is in dim 5
zgemv4D = @(A,b) sum( bsxfun(@times , A , permute(b,[1:4 6 5]) ) ,6);
%%







for iter = 1:maxNumIter
    %   ____
    %  |
    %  |____
    %  |
    %  |____ - S    S T E P
    
    
    
    %% E-S   Source Images
    
    % {J x 1} x [F x L]
    u = cellfun(@(Kj) W(:,Kj) * H(Kj,:), Kj, 'uniformoutput', false);
    
    % [F x L x 1 x J] dim 4 = J
    u = cat(4,u{:});
    
    % [F x L x 1 x J x I x I] R is now u R, M-S will update
    R = bsxfun(@times,u,R);
    
    % [F x L x N x J x I x I] G on WASPAA
    Q = bsxfun(@times,E,R);
    
    % [F x L x N x 1 x I x I] P on WASPAA
    P = sum(Q,4);
    
    % [F x L x N x 1 x I x I] P + vI
    d = bsxfun(@plus,  P,  bsxfun(@times,v,eyeI) );
    
    % [I x I x F x L x N]
    d = permute(d,[5 6 1:4]);
    
    % {F x L x N} x [I x I] cell, inv
    V = cell(F,L,N);   for ind = 1:F*L*N, V{ind} = d(:,:,ind); end
    
    % [1 x L x N] IS det(V) DON'T FORGET THE MINUS IN E-Z as -log|V|
    logDetV = sum( log(cellfun(@det,V)) );
    
    % {F x L x N x J} x [I x I]
    V = cellfun(@inv,V,'uniformoutput',false);
    
    % [IFLN x I]
    V = cat(1,V{:});
    
    % [F x L x N x 1 x I x I]
    V = permute( reshape(V,I,F,L,N,I),[2 3 4 6 1 5] );
    
    % [F x L x N x J x I x I]
    Y = zeros(F,L,N,J,I,I);
    
    for i=1:I
        % [F x L x N x J x I x I] G * V^-1, G is on Q
        Y(:,:,:,:,:,i) = zgemv4D( Q, V(:,:,:,:,:,i) );
    end
    
    for i=1:I
        % [F x L x N x J x I x I] G * V^-1 * G, (until here Q contained G)
        Q(:,:,:,:,:,i) = zgemv4D( Y, Q(:,:,:,:,:,i) );
    end
    
    % [F x L x N] sum G * V^-1 on sources, tr{PV} = sum(P^T .* V)
    delta = sum(sum( sum(Y,4) .* conj(P) ,5) ,6);
    
    % [F x L x N] tr{P} - tr{P * V^-1 * P}
    delta = sum(P(:,:,:,:,1:I+1:I*I) ,5) - real(delta);
    
    % [F x L x N x J x I]
    Y = zgemv4D(Y,X);
    
    % [F x L x N x J x I x I] u*R - G * V^-1 * G
    Q = bsxfun(@minus, R ,Q );
    
    % [F x L x N x J x I x I] Vs + YY^H
    Q = Q + bsxfun(@times, Y, permute(conj(Y),[1 2 3 4 6 5]) );
    %%
    
    
    
    
    %   ____
    %  |
    %  |____
    %  |
    %  |____ - Z    S T E P
    
    
    
    %% E-Z  Diarisation
    
    % [1 x L x N] X^H * V^-1 * X, sum I at dim 5, F at dim 1
    d = sum( sum( bsxfun(@times, conj(X), zgemv4D(V,X) ) ) ,5);
    
    % [N x L] both terms ARE MINUS, real
    d = permute( real( -logDetV -d ) , [3 2 1] );
    
    % cell
    
    % [N x L] subtract max on N
    d = bsxfun(@minus, d, max(d) );
    
    % [N x L] subtract log(sumN(exp(d)))
    d = exp(bsxfun(@minus,d, log(sum(exp(d)))));
    
    % {L} x [N x 1] instantaneous
    iZ = cell(L,1);    for l=1:L, iZ{l} = d(:,l); end
    
    % {L} x [N x 1]
    fZ = [ {permute(Z(:,1,:),[3 2 1]) .* iZ{1}} ;  cell(L-1,1)];
    
    % forward pass
    
    for l=2:L
        % {L} x [N x 1] update forward prob.
        fZ(l) = cellfun(@(iZ,fZ) normalize(iZ .* (T * fZ)), iZ(l),fZ(l-1), 'uniformoutput', false);
    end
    
    % {L} x [N x 1] initialise backward
    bZ = [cell(L-1,1) ; fZ(L)];
    
    % backward pass
    
    for l=L-1:-1:1
        % {L} x [N x 1] update backward prob.
        bZ(l) = cellfun(@(iZ,bZ) normalize(T' * (iZ .* bZ)), iZ(l),bZ(l+1), 'uniformoutput', false);
    end
    
    % marginal of Z
    
    % {L} x [N x 1] marginal posterior prob. of Z
    Z = cellfun(@(fZ,bZ) normalize(fZ .* bZ), fZ, bZ, 'uniformoutput', false);
    
    % [1 x L x N] cast in array
    Z = permute( cat(2,Z{:}) , [3 2 1] );
    %%
    
    
    
    
    %  |\  /|
    %  | \/ | - S    S T E P
    
    
    
    %% Update NMF then R (Q is needed on both)
    
    % [F x L x 1 x J x I x I] sum N
    Q = sum(bsxfun(@times,Z,Q),3);
    
    % [F x 1 x 1 x J x I x I] divide by u
    R = sum( bsxfun(@rdivide,Q,u) ,2) / L;
    
    % [F x 1 x 1 x J x I x I] symtricize
    R = .5 * (R + permute( conj(R),[1 2 3 4 6 5] ))  +  1e-7 * eyeI;
    
    
    % [I x I x F x 1 x 1 x J] use u as tmp, don't alter R, as E-S expects R
    u = permute(R,[5 6 1:4]);
    
    % {F x J} x [I x I] d as tmp
    d = cell(F,J);   for ind = 1:F*J,  d{ind} = u(:,:,ind);  end
    
    % {F x J} x [I x I]
    d = cellfun(@inv,d,'uniformoutput',false);
    
    % [F x 1 x 1 x J x I x I]
    d = permute( reshape(cat(1,d{:}),I,F,J,I), [2 5 6 3 1 4] );
    
    % [F x L x 1 x J] tr{R^-1 * Q} / I              DIVIDE BY I, TAKE REAL
    d = real( sum(sum( bsxfun(@times,d,conj(Q)) ,5),6))/I;
    
    % NMF solve
    for j=1:J
        [W(:,Kj{j}), H(Kj{j},:)]   = nmf_is( d(:,:,j) , 1, W(:,Kj{j})  ,  H(Kj{j},:) );
    end
    %%
    
    
    
    
    %  |\  /|
    %  | \/ | - X    S T E P
    
    
    
    %% M-X noise
    
    % [F x L x N x 1 x I] use d as tmp
    d = sum(Y,4);
    
    % quadratic
    
    % [F x L x N] use Q as tmp
    Q = sum( d.*conj(d) ,5)  +  delta;
    
    % [F x 1] sum L,N
    Q = sum(sum(  bsxfun(@times,Z,Q)  ,2) ,3);
    
    % linear
    
    % [F x L x 1 x 1 x I] avg d
    d = sum( bsxfun(@times,Z,d) ,3);
    
    % [F x 1] X^H * average(d)
    d = sum(sum(   bsxfun(@times,conj(X),d)    ,2),5);
    
    % [F x 1]
    v = ( normX + real( -2*d + Q ) ) / (L*I) + 1e-7;
    %%
    

    
    
    %  |\  /|
    %  | \/ | - Z    S T E P
    
    
    
    %% M-Z transition matrix
    
    % {1 x L-1} x [N x N] joint probability p( Z_{l} , Z_{l-1} )
    d = cellfun(@(bZ,iZ,fZ) bZ .* iZ*fZ' .* T  + 1e-27,  bZ(2:L), iZ(2:L), fZ(1:L-1), 'uniformoutput', false);
    
    % {1 x L-1} x [N x N]
    d = cellfun(@(d) d/sum(d(:)) , d ,'uniformoutput',false);
    
    % [N x N] transition
    T = normalize( sum(cat(3,d{:}),3) );
    
    
    fprintf('pass: %d\n',iter);
end







%% time-domain source images and .rttm

% [F x L x I x J] avg over diarisation states, permute FxLxNxJxI
Y = permute( sum(bsxfun(@times,Y,Z),3) , [1 2 5 4 3] );

y = zeros(M,I,J);

for j=1:J
    y(:,:,j) = transpose( istft_multi( Y(:,:,:,j) , M ) );
end

% [J x N] permute from [1 x 1 x N x J]
permute(E,[4 3 1 2]);

% write diarization output in output.rttm 
z2rttm(strcat('diarization.rttm') ,permute(Z,[3 2 1]),T,permute(E,[4 3 1 2]),M,fs,stft_win_len);
