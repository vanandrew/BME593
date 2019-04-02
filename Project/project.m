clc;
clear;
close all;

% load the data
load('project_data.mat');

% Size of the region of interest (unit: mm)
L = 0.06144;
% Number of pixels in each direction
npixels = 256;
% Pixel size
pixel_size = L/npixels;

% Nubmer of views
nviews = 540;
% Angle increment between views (unit: degree)
dtheta = 5/12;
% Views
views = (0:nviews-1)*dtheta;

% Number of rays for each view
nrays = 512;
% Distance between first and last ray (unit pixels)
d = npixels*(nrays-1)/nrays;

% Construct imaging operator (unit: pixels)
A = paralleltomo(npixels, views, nrays, d);
% Rescale A to physical units (unit: mm)
A = A*pixel_size;
