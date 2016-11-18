function [imo,mask] = rbfwarp2d( im, ps, pd, varargin )
% Radial base function/Thin-plate spline 2D image warping.
% [imo,mask] = rbfwarp2d( im, ps, pd, method)
%   input:
%       im: image 2d matrix
%       ps: 2d source landmark [n*2]
%       pd: 2d destin landmark [n*2]
%       method:
%         'gau',r  - for Gaussian function   ko = exp(-|pi-pj|/r.^2);
%         'thin'   - for Thin plate function ko = (|pi-pj|^2) * log(|pi-pj|^2)
%   output:
%       imo  : output matrix
%       mask : mask for output matrix, 0/1 means out/in the border
%
%   Bookstein, F. L. 
%   "Principal Warps: Thin Plate Splines and the Decomposition of Deformations."
%   IEEE Trans. Pattern Anal. Mach. Intell. 11, 567-585, 1989. 
%
%   Code by WangLin
%   2015-11-5
%   wanglin193@hotmail.com

num_required_parameters = 3;
if nargin < num_required_parameters
    help rbfwarp2d.m
    return;
end

% initialize default parameters
[imh,imw,imc] = size(im);
r = 0.1*imw;
imo = zeros(imh,imw,imc);
mask = zeros(imh,imw);

% parse parameters
if nargin > num_required_parameters
    iVarargin = 1;
    while iVarargin <= nargin - num_required_parameters
        switch lower(varargin{iVarargin})
            case 'thin'
                method = 't';
            case 'gau'
                method = 'g';
                r = varargin{iVarargin+1};
                iVarargin = iVarargin + 1;
        end
        iVarargin = iVarargin + 1;
    end
end

%% Training w with L
nump = size(pd,1);
num_center = size(ps,1);
K=zeros(nump,num_center);

for i=1:num_center
    %Inverse warping from destination!
    dx = ones(nump,1)*ps(i,:)-pd; 
    K(:,i) = sum(dx.^2,2);
end

if( strcmpi(method,'g') )
    K = rbf(K,r);
elseif( strcmpi(method,'t') )
    K = ThinPlate(K);
end

% P = [1,xp,yp] where (xp,yp) are n landmark points (nx2)
P = [ones(num_center,1),pd];
% L = [ K  P;
%       P' 0 ]
L = [K,P;P',zeros(3,3)];
% Y = [x,y;
%      0,0]; (n+3)x2
Y = [ps;zeros(3,2)];
%w = inv(L)*Y;

c = cond(L)


w = L\Y;

w2 = linsolve(L,Y)

csvwrite('w2.csv',w2);
csvwrite('L.csv',L);
csvwrite('Y.csv',Y);
csvwrite('w.csv',w);


%% Using w
[x,y] = meshgrid(1:imw,1:imh);
pt = [x(:), y(:)];

nump = size(pt,1);
Kp = zeros(nump,num_center);
for i=1:num_center
    dx = ones(nump,1)*ps(i,:)-pt;
    Kp(:,i) = sum(dx.^2,2);
end
if( strcmpi(method,'g') )
    Kp = rbf(Kp,r);
elseif( strcmpi(method,'t') )
    Kp = ThinPlate(Kp);    
end

L2 = [Kp,ones(nump,1),pt];
ptall = L2*w;

%csvwrite('L.csv',ret);

%reshape to 2d image
%xd = reshape( ptall(:,1),imh,imw );
%yd = reshape( ptall(:,2),imh,imw );

csvwrite('im.csv',im(:,:,1));
csvwrite('ptall.csv',ptall);


for i = 1:imc
    imt= interp2( single(im(:,:,i)),ptall(:,1),ptall(:,2),'linear');
    imo(:,:,i) = reshape(uint8(imt),imh,imw);
end


mask = ~isnan(imt);

%ret = zeros(imh*imh,6);
%m = 1;
%for i = 1:imc
%    for j=1:imh
%        for k=1:imw
%            imt= interp2( single(im(:,:,i)),xd(j,k),yd(j,k),'spline');
%            ret(m,1) = j;
%            ret(m,2) = k;
%            ret(m,3) = xd(j,k);
%            ret(m,4) = yd(j,k);
%            ret(m,5) = im(j,k);
%            ret(m,6) = imt;
%            m = m +1;
%        end
%    end
%end


mask = ~isnan(imt);
end

function ko = rbf(d,r) 
    ko = exp(-d/r.^2);
end

function ko = ThinPlate(ri)
% k=(r^2) * log(r^2)
    r1i = ri;
    r1i((ri==0))=realmin; % Avoid log(0)=inf
    ko = (ri).*log(r1i);
end
