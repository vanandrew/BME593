close all;
clear all;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle





% Assemble the X-ray tomography matrix, the true data, and true image
[K, d, m_true] = paralleltomo(N, theta, p);

subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');

% Remove possibly 0 rows from K and d
[K, d] = purge_rows(K, d);

% A) Reconstruct m using ART. Report convergence history (i.e. residual and
% error norms at each iteration)

% B) Reconstruct m using SART. Report convergence history and compare with
% ART

% C) Reconstruct m using SIRT. Report convergence history and compare with
% ART and SART

% D) Consider the case with noisy data. Reconstruct m using ART, SART,
% SIRT. Report convergence history and discuss what you observed.
noise_level = 0.01; % noise level.
noise_std = noise_level*norm(d,'inf');
d = d + noise_std*randn(size(d));

% E) Implement Morozov discrepancy principle as stopping criterion for ART