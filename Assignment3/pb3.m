clc;
clear;
close all;

step = 1e-2;
x = -1:step:1;
y = -1:step:1;
[xx, yy] = meshgrid(x,y);
nx = numel(x);
ny = numel(y);
f = zeros(nx,ny);
g = zeros(nx,ny,2);
H = zeros(nx,ny,2,2);
for i=1:nx
    for j=1:ny
        [f(i,j),g(i,j,:),H(i,j,:,:)] = objectiveFunction([x(i);y(j)]);
    end
end
[gy, gx] = gradient(f,step);
a = g(:,:,1) - gx; 
b = g(:,:,2) - gy;
quiver(xx,yy,a,b);

