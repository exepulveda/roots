function [D, asize, bb] = decmat(scale,bsize,whatsize,pack)
%
% Decimation matrix with Gaussian filter
%
% [D asize bb] = decmat(scale,bsize,whatsize,shifts)
%
% scale ... scalar; scale factor
% bsize ... 1x2 vector; size of the input or the output depending on whatsize 
% whatsize ... 'i' input size, 'o' output size is specified in bsize
% pack ... cell array {[y1 x1] [y2 x2] ...}; predetermined shifts (in the 
% scale of the LR image); 
%
% output arg.:
% D ... decimation matrix (cell array if shifts is nonempty)
% asize ... size of the input or output (opposite to bsize)
%
% note: D is constructed in such a way that only the inner part of convolution 
% during the decimation operation is returned where both the filter and 
% image are fully defined 
%

% % written by Filip Sroubek (C) 2006

if length(scale) == 1
  scale = [scale scale];
end

if nargin < 4
  pack = {[0 0]};
end

shifts = [pack{:}];
shifts = [scale(1)*shifts(1:2:end); scale(2)*shifts(2:2:end)];

if lower(whatsize(1)) == 'i'
  isize = {bsize(1) bsize(2)};
  osize = {[] []};
else
  osize = {bsize(1) bsize(2)};
  isize = {[] []};
end

% 1D decimation matrices in X and Y directions
[Y, limy] = gaussmask(scale(1),osize{1},isize{1},shifts(1,:));
[X, limx] = gaussmask(scale(2),osize{2},isize{2},shifts(2,:));
as = [size(Y,1), size(X,1)]/size(shifts,2);
for k = 0:size(shifts,2)-1
    % construct 2D decimation matrix from two 1D matrices X and Y
  D{k+1} = kron(X([1:as(2)]+k*as(2),:),Y([1:as(1)]+k*as(1),:));
end
if nargin < 4 
  D = D{1};
end
if lower(whatsize(1)) == 'i'
  asize = [size(Y,1), size(X,1)]/size(shifts,2);
else
  asize = [size(Y,2), size(X,2)];
end
bb = [limy, limy];

