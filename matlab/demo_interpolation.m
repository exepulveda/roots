clear all, close all

% load image
original = imread('im.tiff');

% load correspondences 
ps = dlmread('ps.txt'); % image points
pd = dlmread('pd.txt'); % reference points

% interpolate
im = rgb2gray(original);
w = size(im,2);
rectified = rbfwarp2d( im, ps, pd,'gau',10*w);

% plot
subplot(1,2,1)
imshow(original)
title('Orininal image');
hold on
plot( ps(:,1),ps(:,2),'r*' );

subplot(1,2,2)
imshow(uint8(rectified));
title('Thin-plate warping');
hold on
plot( ps(:,1),ps(:,2),'r*' );
plot( pd(:,1),pd(:,2),'gO' );
