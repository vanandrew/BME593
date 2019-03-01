clc;
clear;
close all;
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

figure;
subplot(121);
imagesc(reshape(m_true, N, N));
title('True image');
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');

% Remove possibly 0 rows from K, and d
[K, d] = purge_rows(K, d);

% Rescale K, and d so that the l2 norm of each row is 1
s = sqrt(sum(K.*K, 2));
K = spdiags(1./s,0)*K;
d = spdiags(1./s,0)*d;

% A) Reconstruct m using EM. Report D_KL(d || K*m) and ||m - m_true||

% initialize variables
iterations = 100;
m_recon = ones(size(m_true,1),101);
KL_divergence = zeros(iterations,1);
norm_error = zeros(iterations,1); 
sp = sum(K,1);
SinvKT = diag(sp.^-1)*K';
for i=1:iterations
    fprintf('Iteration #: %d\n', i);
    m_recon(:,i+1) = m_recon(:,i).*SinvKT*(d./(K*m_recon(:,i)));
    KL_divergence(i) = sum(d.*log(d./(K*m_recon(:,i+1))),1);
    norm_error(i) = norm(m_true - m_recon(:,i+1));
end

% plot KL Divergence and norm error
figure;
semilogy(1:100,KL_divergence); title('KL Divergence on m_{recon}');
ylabel('D_{KL}(d||Km^{(j)})'); xlabel('Iteration #');
figure;
semilogy(1:100,norm_error); title('Euclidean Error Norm on m_{recon}');
ylabel('||m_{true} - m^{(j)}||'); xlabel('Iteration #');

% plot true image vs. reconstructed
figure;
subplot(121);
imagesc(reshape(m_true, N, N)); title('True Image');
subplot(122);
imagesc(reshape(m_recon(:,101), N, N)); title('Reconstructed Image (EM)');

% B) Consider the noisy data case, i.e d is a realization of a Poisson
% distributed random vector.
% Design an appropriate stopping criterion for EM and reconstruct m
dn = poissrnd(K*m_true);

% initialize variables
iterations = 100;
mn_recon = ones(size(m_true,1),101);
KL_divergence_n = zeros(iterations,1);
norm_error_n = zeros(iterations,1); 
for i=1:iterations
    fprintf('Iteration #: %d\n', i);
    mn_recon(:,i+1) = mn_recon(:,i).*SinvKT*(dn./(K*mn_recon(:,i)));
    KL_sum = dn.*log(dn./(K*mn_recon(:,i+1))); KL_sum_rn = KL_sum(~isnan(KL_sum)); % remove nans
    KL_divergence_n(i) = sum(KL_sum_rn,1);
    norm_error_n(i) = norm(m_true - mn_recon(:,i+1));
end

% plot KL Divergence and norm error
figure;
semilogy(1:100,KL_divergence_n); title('KL Divergence on m_{recon} w/ noise');
ylabel('D_{KL}(d||Km^{(j)})'); xlabel('Iteration #');
figure;
semilogy(1:100,norm_error_n); title('Euclidean Error Norm on m_{recon} w/ noise');
ylabel('||m_{true} - m^{(j)}||'); xlabel('Iteration #');

% plot true image vs. reconstructed
figure;
subplot(121);
imagesc(reshape(m_true, N, N)); title('True Image');
subplot(122);
imagesc(reshape(mn_recon(:,101), N, N)); title('Reconstructed Image (EM)');

