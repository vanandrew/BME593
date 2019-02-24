close all;
clear all;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle

% Assemble the X-ray tomography matrix, the true data, and true image
K = paralleltomo(N, theta, p);

% Generate synthetic data
m_true = phantomgallery('smooth', N);
m_true = m_true(:);
d = K*m_true;

subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');

% Remove possibly 0 rows from K, and d
[K, d] = purge_rows(K, d);

% Rescale K, and d so that the l2 norm of each row is 1
s = sqrt( sum(K.*K, 2) );
K = spdiags(1./s,0)*K;
d = spdiags(1./s,0)*d;

% A) Reconstruct m using EM. Report D_KL(d || K*m) and ||m - m_true||

% B) Consider the noisy data case, i.e d is a realization of a Poisson
% distributed random vector.
% Design an appropriate stopping criterion for EM and reconstruct m
d = poissrnd(K*m_true);

