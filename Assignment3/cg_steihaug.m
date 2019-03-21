function [x, i] = cg_steihaug(A,b,maxiter,rtol,x)
%   Conjugate Gradient Method.
%   X = MYCG(A,B,maxiter,tol,x0) solves the system of linear equations A*X=B
%   for X. The N-by-N coefficient matrix A must be symmetric and the right
%   hand side column vector B must have length N.

r = A*x-b;
d = -r;
rsold = r'*r;

tol = sqrt(rsold)*rtol;

for i=1:maxiter
    Ad = A*d;
    den = (d'*Ad);
    if den <= 0 %Steihaug check for negative curvature
        break;
    end
    alpha = rsold/den;
    x = x+alpha*d;
    r = r+alpha*Ad;
    rsnew = r'*r;
    if sqrt(rsnew)<tol
    	break;
    end
    beta = rsnew/rsold;
    d = -r+beta*d;
    rsold = rsnew;
end