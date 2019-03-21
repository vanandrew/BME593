function [f, g, H] = objectiveFunction(x)

n = size(x,1);
h = 1/n;
K = zeros(n,n);
b = zeros(n,1);

for i = 1:n
    b(i) = i*h;
    for j=1:n
        K(i,j) = h*exp( - .5*h^2*(i-j)^2);
    end
end

r = K*x + exp(x) + b;
J = K + diag( exp(x) );

f = .5*(r'*r) + .5*(x'*x);

g = J'*r + x;

H = J'*J +  diag(exp(x).*r) + eye(n);

end