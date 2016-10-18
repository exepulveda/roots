clear all
close all

addpath images

%imageName='1.jpg';
%imageName='3.tiff';
%imageName='4.tiff';
imageName='6.tiff';

%imageName='2.tiff';

ds = 15;


I_original = imread(imageName);

padding = 3;
I_pad = I_original(padding +1:end-padding, padding +1:end-padding,:);

I= rgb2gray(I_pad);
im = I;


subplot(1,2,1)
imshow(uint8(im))
title('Orininal image with base landmarks');
hold on



%% Fit Circles
tic
[h,w]=size(im);

% E
E = im(:, (2/3)*w:w);
E = imgaussfilt(E,2);

eCircle = struct();
[eCircle.c, eCircle.r, eCircle.base] = circledetection(E);

eCircle.c = eCircle.c + [(2/3)*w, 0];
eCircle.base = eCircle.base + repmat([(2/3)*w, 0], 3,1);

% W
W = im(:, 1:(1/3)*w);
W = imgaussfilt(W,2);

wCircle = struct();
[wCircle.c, wCircle.r, wCircle.base] = circledetection(W);

% N
N = im(1:(1/3)*h, :);
N = imgaussfilt(N,2);

nCircle = struct();
[nCircle.c, nCircle.r, nCircle.base] = circledetection(N);

% S
S = im((2/3)*h:h, :);
S = imgaussfilt(S,2);

sCircle = struct();
[sCircle.c, sCircle.r, sCircle.base] = circledetection(S);

sCircle.c = sCircle.c + [0, (2/3)*h];
sCircle.base = sCircle.base + repmat([0, (2/3)*h], 3,1);
toc

% Corners
%%
imsize=[h,w];
a = findcorner(wCircle, nCircle, imsize);
b = findcorner(eCircle, nCircle, imsize);
c = findcorner(eCircle, sCircle, imsize);
d = findcorner(wCircle, sCircle, imsize);

xN = linspace(0, w-1,ds );
xS = linspace(0, w-1,ds );

yW = linspace(0, h-1,ds );
yE = linspace(0, h-1,ds );


plot_circle(nCircle.c, nCircle.r);
plot(nCircle.base(:,1), nCircle.base(:,2), '*b')

plot_circle(sCircle.c, sCircle.r);
plot(sCircle.base(:,1), sCircle.base(:,2), '*b')

plot_circle(wCircle.c, wCircle.r);
plot(wCircle.base(:,1), wCircle.base(:,2), '*b')

plot_circle(eCircle.c, eCircle.r);
plot(eCircle.base(:,1), eCircle.base(:,2), '*b')


plot(a(1), a(2), '*g')
plot(b(1), b(2), '*g')
plot(c(1), c(2), '*g')
plot(d(1), d(2), '*g')


% sample points

ptsN = zeros(ds,2);
ptsS = zeros(ds,2);
ptsW = zeros(ds,2);
ptsE = zeros(ds,2);


ptsN(:,1) = linspace(a(1), b(1), ds);
ptsN(:,2) = circevalx(ptsN(:,1), nCircle, imsize);

ptsS(:,1) = linspace(d(1), c(1), ds);
ptsS(:,2) = circevalx(ptsS(:,1), sCircle, imsize);

ptsW(:,2) = linspace(a(2), d(2), ds);
ptsW(:,1) = circevaly(ptsW(:,2), wCircle, imsize);

ptsE(:,2) = linspace(b(2), c(2), ds);
ptsE(:,1) = circevaly(ptsE(:,2), eCircle, imsize);


plot(ptsN(:,1), ptsN(:,2), 'oy');
plot(ptsS(:,1), ptsS(:,2), 'oy');
plot(ptsW(:,1), ptsW(:,2), 'oy');
plot(ptsE(:,1), ptsE(:,2), 'oy');


% N S W E
pd =1+[xN(2:end-1)',       zeros(ds-2,1)
      xS(2:end-1)',       (h-1)*ones(ds-2,1)
     zeros(ds,1),     yW'
     (w-1)*ones(ds,1),   yE'];

ps = [ptsN(2:end-1,:); ptsS(2:end-1,:); ptsW; ptsE];


% Thin plate warping
%[imo1,mask1] = rbfwarp2d( im, ps, pd,'thin');
[imo1,mask1] = rbfwarp2d( im, ps, pd,'gau',10*w);


subplot(1,2,2)
imshow(uint8(imo1));
title('Thin-plate warping');
hold on
plot( ps(:,1),ps(:,2),'r*' );
plot( pd(:,1),pd(:,2),'gO' );
