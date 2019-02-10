clc;
clear;
close all;

% Create 200 points (x and x' have the same domain)
x = linspace(0, 1, 200);

% Create the blurring operator k
k = @(m, n) (1/numel(m))*...
            (0.2^-2)*...
            max(zeros(numel(m),numel(n)),0.2-abs(repmat(m',1,numel(n))-repmat(n,numel(m),1)));

% Now create m_true function
m_true = @(c) ...
    ((0.1<c & c<0.25)*0.75 + ...
    (0.3<c & c<0.32)*0.25 + ...
    (0.5<c & c<1).*sin(2*pi*c))';

% Add le gaussian noise with 0 mean, 0.1 variance
n = normrnd(0,sqrt(0.1),numel(x),1);

% Now calculate d and dn
d = k(x,x)*m_true(x);
dn = d + n;

% plot d and d with noise
figure; hold on; a = plot(d); b = plot(dn);
xlabel('x'); ylabel('d'); title('d and d w/ noise');
legend([a,b],{'d','d w/noise'})

% do svd on k
[U,S,V] = svd(k(x,x));

%% Part (a)

% calculate TSVD filters (alpha = 0.0001, 0.001, 0.1, 1)
alpha = [0.0001, 0.001, 0.1, 1];
[r_k, c_k] = size(k(x,x)); % get size of k
k_inv = zeros(r_k, c_k, numel(alpha)); % initial storage for k_inv
e_m = numel(m_true(x)); % get size of m
m_recon = zeros(e_m, numel(alpha)); % initial storage for m_recon
for i=1:numel(alpha)
    % Find eigenvalues greater than alpha and use as a mask
    mask = diag(S.^2) > alpha(i);
    
    % Now truncate with mask and invert the singular values
    S_inv = diag(diag(1./S).*mask);
    
    % Now find the inverse
    k_inv(:,:,i) = V*S_inv*U';
    
    % Reconstruct
    m_recon(:,i) = k_inv(:,:,i)*dn;
end

% plot everything
figure('Position', [400, 100, 900, 900]); subplot(numel(alpha)+1,1,1); % subplot with m_true on top
plot(x,m_true(x)); % plot m_true
xlim([0,1]); ylim([-1,1]); title('m_{true}'); xlabel('x'); ylabel('m');
for i=1:numel(alpha) % plot reconstructed m for each alpha
    subplot(numel(alpha)+1,1,i+1);
    plot(x,m_recon(:,i));
    xlim([0,1]); ylim([-1,1]);
    title(['m_{recon}, \alpha = ', num2str(alpha(i))]); xlabel('x'); ylabel('m');
end

%% Part (b)

% calculate Tikhinov filters (alpha = 0.0001, 0.001, 0.1, 1)
k_inv_tk = zeros(r_k, c_k, numel(alpha)); % initial storage for k_inv
m_recon_tk = zeros(e_m, numel(alpha)); % initial storage for m_recon
for i=1:numel(alpha)
    % Create Tikhinov mask
    mask = diag(S.^2)./(diag(S.^2) + alpha(i));
    
    % Now truncate with mask and invert the singular values
    S_inv = diag(diag(1./S).*mask);
    
    % Now find the inverse
    k_inv_tk(:,:,i) = V*S_inv*U';
    
    % Reconstruct
    m_recon_tk(:,i) = k_inv_tk(:,:,i)*dn;
end

% plot everything
figure('Position', [400, 100, 900, 900]); subplot(numel(alpha)+1,1,1); % subplot with m_true on top
plot(x,m_true(x)); % plot m_true
xlim([0,1]); ylim([-1,1]); title('m_{true}'); xlabel('x'); ylabel('m');
for i=1:numel(alpha) % plot reconstructed m for each alpha
    subplot(numel(alpha)+1,1,i+1);
    plot(x,m_recon_tk(:,i));
    xlim([0,1]); ylim([-1,1]);
    title(['m_{recon}, \alpha = ', num2str(alpha(i))]); xlabel('x'); ylabel('m');
end

%% part (c)

% calculate misfit and regularization values
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1, 1e1, 1e2, 1e3, 1e4];
misfit = zeros(numel(alpha_list),1);
reg = zeros(numel(alpha_list),1);
for i=1:numel(alpha_list)
    % calculate m_alpha
    m_alpha = (k(x,x)'*k(x,x) + alpha_list(i) * eye(numel(diag(S))))\(k(x,x)'*dn);
    plot(m_alpha)
    
    % calculate misfit and regularization term for alpha
    misfit(i) = norm(k(x,x)*m_alpha - dn);
    reg(i) = norm(m_alpha);
end

% plot L-curve
figure; loglog(misfit, reg); hold on;
loglog(misfit(6), reg(6), 'ro', 'Linewidth', 3); % circle optimal value
xlabel('||K*m - d||'); ylabel('||m||'); title('L-curve');

%% part (d)

% calculate norm of noise
delta = norm(n);

% plot morozov
figure; loglog(alpha_list, misfit); hold on;
loglog(alpha_list, delta*ones(size(alpha_list)), 'Linewidth', 1); % plot delta
loglog(alpha_list(6), misfit(6), 'ro', 'Linewidth', 3); % circle optimal value
xlabel('\alpha'); ylabel('||K*m - d||'); title('Morozov');

%% part (e)


%% part(f)