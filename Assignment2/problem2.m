clc;
clear;
close all;

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

% Remove possibly 0 rows from K and d (get index as well)
[K, d, idx] = purge_rows(K, d);

% A) Reconstruct m using ART. Report convergence history (i.e. residual and
% error norms at each iteration)

% initalize arrays to save
iterations = 1000;

% precompute the norm of each row
K_norms = full(sum(K.^2,2));

% run ART (See ART.cpp file)
tic;
m_art = ART(K',K_norms,d);
toc;

% calculate residual and error
residual_art = sum((repmat(d,1,1000) - K*m_art).^2,1);
error_art = sum((repmat(m_true,1,1000) - m_art).^2,1);

% plot stuff
figure; imagesc(reshape(m_art(:,1000),64,64)); title('Reconstruction (ART)');
figure; semilogy(1:1000,residual_art); title('||Residual|| (ART)');
xlabel('Iteration #'); ylabel('||d - Km^{(j)}||');
figure; semilogy(1:1000,error_art); title('||Error|| (ART)');
xlabel('Iteration #'); ylabel('||m_{true} - m^{(j)}||');

% B) Reconstruct m using SART. Report convergence history and compare with
% ART

% run SART (See SART.cpp file)
tic;
m_sart = SART(K',K_norms,d,idx);
toc;

% calculate residual and error
residual_sart = sum((repmat(d,1,1000) - K*m_sart).^2,1);
error_sart = sum((repmat(m_true,1,1000) - m_sart).^2,1);

% plot stuff
figure; imagesc(reshape(m_sart(:,1000),64,64)); title('Reconstruction (SART)');
figure; semilogy(1:1000,residual_sart); title('||Residual|| (SART)');
xlabel('Iteration #'); ylabel('||d - Km^{(j)}||');
figure; semilogy(1:1000,error_sart); title('||Error|| (SART)');
xlabel('Iteration #'); ylabel('||m_{true} - m^{(j)}||');

% C) Reconstruct m using SIRT. Report convergence history and compare with
% ART and SART

% run SIRT (See SIRT.cpp file)
tic;
m_sirt = SIRT(K',K_norms,d);
toc;

% calculate residual and error
residual_sirt = sum((repmat(d,1,1000) - K*m_sirt).^2,1);
error_sirt = sum((repmat(m_true,1,1000) - m_sirt).^2,1);

% plot stuff
figure; imagesc(reshape(m_sirt(:,1000),64,64)); title('Reconstruction (SIRT)');
figure; semilogy(1:1000,residual_sirt); title('||Residual|| (SIRT)');
xlabel('Iteration #'); ylabel('||d - Km^{(j)}||');
figure; semilogy(1:1000,error_sirt); title('||Error|| (SIRT)');
xlabel('Iteration #'); ylabel('||m_{true} - m^{(j)}||');

% D) Consider the case with noisy data. Reconstruct m using ART, SART,
% SIRT. Report convergence history and discuss what you observed.
noise_level = 0.01; % noise level.
noise_std = noise_level*norm(d,'inf');
dn = d + noise_std*randn(size(d));

% Run ART, SART, and SIRT
tic;
mn_art = ART(K',K_norms,dn);
mn_sart = SART(K',K_norms,dn,idx);
mn_sirt = SIRT(K',K_norms,dn);
toc;

% calculate residuals and errors
residualn_art = sum((repmat(dn,1,1000) - K*mn_art).^2,1);
errorn_art = sum((repmat(m_true,1,1000) - mn_art).^2,1);
residualn_sart = sum((repmat(dn,1,1000) - K*mn_sart).^2,1);
errorn_sart = sum((repmat(m_true,1,1000) - mn_sart).^2,1);
residualn_sirt = sum((repmat(dn,1,1000) - K*mn_sirt).^2,1);
errorn_sirt = sum((repmat(m_true,1,1000) - mn_sirt).^2,1);

% plot stuff
figure; imagesc(reshape(mn_art(:,1000),64,64)); title('Reconstruction (ART)');
figure; imagesc(reshape(mn_sart(:,1000),64,64)); title('Reconstruction (SART)');
figure; imagesc(reshape(mn_sirt(:,1000),64,64)); title('Reconstruction (SIRT)');
figure; title('||Residual||'); hold on;
semilogy(1:1000,residualn_art);
semilogy(1:1000,residualn_sart);
semilogy(1:1000,residualn_sirt);
xlabel('Iteration #'); ylabel('||d - Km^{(j)}||');
hold off;
figure; title('||Error||'); hold on;
semilogy(1:1000,errorn_art);
semilogy(1:1000,errorn_sart);
semilogy(1:1000,errorn_sirt);
xlabel('Iteration #'); ylabel('||m_{true} - m^{(j)}||');
hold off;

% E) Implement Morozov discrepancy principle as stopping criterion for ART
