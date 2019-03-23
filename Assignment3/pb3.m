clc;
clear;
close all;

% settings
n = 250;
iterations = 50;
backtrack_iter = 100;
rtol = 10^-8;
c = 1e-5;

% set initial x
x = zeros(n,1);

for i=1:iterations
    % Get f,g and h
    [f,g,H] = objectiveFunction(x);
    
    % save initital gradient
    if i == 1
       g0 = g; 
    end
    
    % break if <= tolerance
    if norm(g)/norm(g0) <= rtol
        break;
    end
    
    % get pk using cg_steihaug
    %eta = 0.5;
    %eta = min([0.5, sqrt(norm(g)/norm(g0))]);
    eta = min([0.5, norm(g)/norm(g0)]);
    [pk, si] = cg_steihaug(H,-g,1000,eta,zeros(n,1));
    
    % do armijo backtracking
    alpha = 1;
    for j=1:backtrack_iter
        [fprop,~,~] = objectiveFunction(x+alpha*pk);
        if fprop <= f + c*alpha*g'*pk
            break;
        else
            alpha = alpha/2;
        end
    end
    
    % update x
    x = x + alpha*pk;
    plot(x);
end