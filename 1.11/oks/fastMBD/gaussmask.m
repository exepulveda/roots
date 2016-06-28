function [G bb] = gaussmask(scale,count,tsize,initpos)
%
% Gaussian 1D mask 
%
% G = gaussmask(scale,count,tsize,initpos)
% generates gaussian filter that is appropriate 
% for downscaling with a factor "scale"
%
% bb ... limits (as indices): size of G minus size of the blur
%
% scale ... downscaling factor
% count ... size of the output vector (number of rows in G)
% tsize ... size of the input vector (number of columns in G)
% initpos ... shift of the mask center (default is 0)
% if count or tsize is empty it will be calculated to assure "valid-support"
% convolution.
%
% sigma = gaussmask(sigma) ... set sigma (in the LR scale) for subsequent 
%                       calls to gaussmask; dafault sigma is 0.3415
%
% sigma = gaussmask; ... return current sigma
%
% note: similar to gaussian2D but using erf() to minimize the aliasing
% effect and that after downscaling and subsequent upscaling the image 
% will contain only a few artefacts   

% store sigma for subsequent calls
persistent sigma_in_lrscale

% explanation of value sigma in the f-domain
%% at this frequency the spectrum of the filter should approach zero
%fmax = 1/scale;
%% var of the gaussian filter that will have power 1e-1 at fmax
%% idealy the power should be 0 but this would imply sigma=0
%sigma = sqrt(-log(1e-1)/(2*pi^2*fmax^2));
% the above is equivalent to constant sigma = 0.3415 in the LR scale 

if nargin == 1
  sigma_in_lrscale = scale;
end
if isempty(sigma_in_lrscale) 
  sigma_in_lrscale = 0.3415; 
end
if nargin <= 1
  G = sigma_in_lrscale;
  return;
end

sigma = sigma_in_lrscale*scale;

if nargin < 4 
  initpos = [0];
end

shift = [];
for ip = initpos
  if isempty(count)
    shift = [shift, ip + scale/2 + [1:ceil(tsize/scale)]*scale];
  else
    shift = [shift, ip + scale/2 + [1:count]*scale];
  end
end

m = mod(shift(:),1);
n = floor(shift(:)/1);
P = length(shift);
% set size so that at most 10% of energy is missing
% idealy it should be 0% but then the size is infinit
lsize = max(floor(erfcinv(1e-1)*sqrt(2)*sigma-m));
rsize = max(ceil(erfcinv(1e-1)*sqrt(2)*sigma+m));
x = repmat([-lsize:rsize],P,1) - repmat(m,1,rsize+lsize+1);
xlim = cat(3,x-1, x)/(sigma*sqrt(2));
g = diff(0.5*erf(xlim),1,3);
g = g./repmat(sum(g,2),1,size(g,2));
j = repmat([1:rsize+lsize+1],P,1)+repmat(n-lsize-1,1,rsize+lsize+1);
j = j - min(j(:)) + 1;
i = repmat([1:P]',1,rsize+lsize+1);

if nargin == 2 || isempty(tsize)
  tsize = max(j(:));
  G = sparse(i,j,g,P,tsize);
else
  M = j>=1 & j<=tsize;
  f = sum(M,2) == rsize+lsize+1;
  F = reshape(f,P/length(initpos),length(initpos));
  CF = cumsum(F);
  F(CF > min(CF(end,:))) = 0;
  f = F(:);
  G = sparse(i(M),j(M),g(M),P,tsize);
  G = G(f,:);
end

% calculate limits
bb = [1+lsize, tsize-rsize];
