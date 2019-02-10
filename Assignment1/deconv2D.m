close all
clear all

% Import image
I = imread('circle.png');
m_true = 146*double(I);

% Get the number of pixels in the vertical and horizontal direction (N1, and N2)
[N1, N2] = size(m_true);
N = N1 * N2;

% Generate x and y axis
x=linspace(0,N2,N2);
y=linspace(0,N1,N1);
[xx,yy] = meshgrid(x,y);

% draw the imported image
figure;
colormap gray;
imagesc(m_true);
title('True image');


% Use different Gaussian blurring in x and y-direction
gamma1 = 5;
C1 = 1 / (sqrt(2*pi)*gamma1);
gamma2 = 12;
C2 = 1 / (sqrt(2*pi)*gamma2);

% blurring operators for x and y directions
K1 = zeros(N1,N1);
K2 = zeros(N2,N2);
for l = 1:N1
    for k = 1:N1
    	K1(l,k) = C1 * exp(-(l-k)^2 / (2 * gamma1^2));
    end
end
for l = 1:N2
    for k = 1:N2
    	K2(l,k) = C2 * exp(-(l-k)^2 / (2 * gamma2^2));
    end
end

% blur the image: first, K2 is applied to each column of I,
% then K1 is applied to each row of the resulting image
Ib = (K2 * (K1 * m_true)')';

% plot blurred image
figure;
colormap gray;
imagesc(Ib);
title('blurred image');

% add nose and plot noisy blurred image
Ibn = Ib + 16 * randn(N1,N2);
figure;
colormap gray;
imagesc(Ibn);
title('blurred noisy image');

% compute Tikhonov reconstruction with regularization
% parameter alpha, i.e. compute m = (K'*K + alpha*I)\(K'*d)

% first construct the right hand side K'*d
K_Ibn = (K2 * (K1 * Ibn)')';

% then set the regularization parameter 
alpha = 1.5e-3;

% now solve the regularized inverse problem to reconstruct the 
% the image using preconditioned conjugate gradients (pcg) to solve the
% system in a matrix-free way using function "apply"

m_alpha = pcg(@(in)apply(in,K1,K2,N1,N2,alpha),K_Ibn(:),1e-6,1500);
figure;
%colormap gray;
imagesc(reshape(m_alpha,N1,N2));
title('Tikhonov reconstruction');