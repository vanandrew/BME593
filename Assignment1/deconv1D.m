clear all
close all

% Number of discretization points
N = 128;
gamma = 0.03;
C = 1 / (sqrt(2*pi)*gamma);

K = zeros(N,N);
h = 1/N;
x = linspace(0,1,N)';

% discrete convolution matrix
for l = 1:N
    for k = 1:N
    	K(l,k) = h * C * exp(-(l-k)^2 * h^2 / (2 * gamma^2));
    end
end

% true image
m = (x > .2).*(x < .3) + sin(4*pi*x).*(x > 0.5) + 0.0 * cos(30*pi*x);

% convolved image
d = K * m;

% noisy data, noise has sigma (standard deviation) = 0.1

dn = d + 0.1 * randn(N,1);
plot(x,d,x,dn,'Linewidth', 2);
legend('data', 'noisy data');

% Tikhonov regularization parameter
alpha = 0.1;

% solve Tikhonov system
m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
% comment out next 3 if you dont want figure
figure;
plot(x,m,x,m_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
legend('exact data', 'Tikhonov reconstruction');

% plot L-curve
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
no = length(alpha_list);
misfit = zeros(no,1);
reg = zeros(no,1);

for k = 1:no
    alpha = alpha_list(k);
    m_alpha = (K'*K + alpha * eye(N))\(K'*dn);
    misfit(k) = norm(K*m_alpha - dn);
    reg(k) = norm(m_alpha);
end

figure;
loglog(misfit, reg, 'Linewidth', 2);
hold on;
loglog(misfit(5), reg(5), 'ro', 'Linewidth', 3);
xlabel('||K*m - d||'); ylabel('||m||');
