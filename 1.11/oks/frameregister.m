addpath fastMBD

%% Read images

% Frame to register
frame=46;
%frame=14;

inames=dir(sprintf('frame-%d/*.jpg',frame));
inames={inames.name};

% Load all images

n = length(inames)-1;
images=cell(1,n);

I0=imread(sprintf('frame-%d/%s',frame,inames{1}));
for i=1:n
    images{i}=imread(sprintf('frame-%d/%s',frame,inames{i+1}));
end

% Show first and last images
%imshowpair(I0,images{end},'montage')


%% Register

fixed = rgb2gray(I0);
H=cell(1,n);

for i=1:n
    H{i}=homography(rgb2gray(images{i}),fixed);
end


%% Warp

% Transform the moving image using the transform estimate from imregtform.
% Use the 'OutputView' option to preserve the world limits and the
% resolution of the fixed image when resampling the moving image.

W=cell(1,n);
R=imref2d(size(I0)); %asumes all images are equal size

for i=1:n
    W{i}=imwarp(images{i},R,H{i},'OutputView',R, 'SmoothEdges', true);
end

% add I0 to W
W{end+1}=I0;

%% Blend

dim=[R.ImageExtentInWorldY,R.ImageExtentInWorldX,length(W)];

r=reshape(cell2mat(cellfun(@(x) x(:,:,1),W,'UniformOutput',false)),dim);
g=reshape(cell2mat(cellfun(@(x) x(:,:,2),W,'UniformOutput',false)),dim);
b=reshape(cell2mat(cellfun(@(x) x(:,:,3),W,'UniformOutput',false)),dim);

f=@median;
r = f(r,3);
g = f(g,3);
b = f(b,3);

B=cat(3,r,g,b);
%imshowpair(I0,B,'montage')

%% MC

[M hr] = MCrestoration({images{:},I0},[20 20]);
M=uint8(M);


[Mw hr] = MCrestoration(W,[20 20]);
Mw=uint8(Mw);

%% Show 2 results
close all
subplot(3,1,1)
imshowpair(I0,B,'montage')
title('registration+median')

subplot(3,1,2)
imshowpair(I0,M,'montage')
title('registration+restoration')

subplot(3,1,3)
imshowpair(I0,Mw,'montage')
title('restoration')
