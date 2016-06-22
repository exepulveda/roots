function [R tr] = gradUsefulness(G,hsize)

% Generate measure of the gradient usefulness as proposed by Xu and Jia,
% ECCV 2010
%
% G .. grayscale image
% hsize ... size of neighborhood (PSF size)

gsize = [size(G,1),size(G,2)];

%[Gx Gy] = gradient(G);
%FGx = fft2(Gx);
%FGy = fft2(Gy);

FDx = fft2([1 -1],gsize(1),gsize(2));
FDy = fft2([1; -1],gsize(1),gsize(2));
FG = fft2(G);
FGx = FDx.*FG;
FGy = FDy.*FG;
Gx = real(ifft2(FGx));
Gy = real(ifft2(FGy));

% hsize should be odd
hsize = 1 + 2*floor((hsize-1)/2)
hshift = zeros(floor(hsize/2)+1); hshift(end) = 1;
PSF = ones(hsize)/prod(hsize);
FNG = fft2(sqrt(Gx.^2+Gy.^2));
M = conj(fft2(hshift,gsize(1),gsize(2))).*...
fft2(PSF,gsize(1),gsize(2));
R = sqrt(real(ifft2(FGy.*M)).^2 + real(ifft2(FGx.*M)).^2)./...
    (real(ifft2(FNG.*M)) + 0.5);

%R = edgetaper(R,PSF);
wx = [(0:hsize(2)-1)/hsize(2), ones(1,gsize(2)-2*hsize(2)), (hsize(2)-1:-1:0)/hsize(2)];
wy = [(0:hsize(1)-1)/hsize(1), ones(1,gsize(1)-2*hsize(1)), (hsize(1)-1:-1:0)/hsize(1)];
R  = R .*(wy.'*wx);



% for groups d1,...,d4 (directions) of gradients
d1 = Gx>=0 & Gy>=0;
d2 = Gx>=0 & Gy<0;
d3 = Gx<0 & Gy<0;
d4 = Gx<0 & Gy>=0;

% the minimum number of pixels to be selected in each group
t = 0.5*sqrt(prod(gsize)*prod(hsize));
M = linspace(min(R(:)),max(R(:)),100);
N = zeros(4,100);
N(1,:) = cumsum(flipud(histc(R(d1),M)));
N(2,:) = cumsum(flipud(histc(R(d2),M)));
N(3,:) = cumsum(flipud(histc(R(d3),M)));
N(4,:) = cumsum(flipud(histc(R(d4),M)));


i = find(sum(N>t)==4,1);
if i == length(M) || isempty(i)
    i = length(M)-1;
end
tr = M(end-i);
%tr = M(i+1);


    

