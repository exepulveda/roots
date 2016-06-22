function [d e] = motionestuw(F,G)

% d = motionest(F,G)
% estimation of subpixel misalignment d = [dy, dx] between F and G
% with upwind scheme
% e ... error of optical flow equation in the image for single translation vector d
% based on the optical flow equation


h = fspecial('gaussian',[11 11],3);
%h = gaussblur([21 21],10);
%h = ones(31,31); h = h/sum(h(:));
F = convn(F,h,'valid');
G = convn(G,h,'valid');

is = [size(F,1),size(F,2),size(F,3)];

%backward
Fx{1} = [ zeros(is(1),1,is(3)), diff(F,1,2)];
Gx{1} = [ zeros(is(1),1,is(3)), diff(G,1,2)];
%forward
Fx{2} = [ diff(F,1,2), zeros(is(1),1,is(3))];
Gx{2} = [ diff(G,1,2), zeros(is(1),1,is(3))];
%central
FGx = 0.25*(Fx{1}+Fx{2}+Gx{1}+Gx{2});

%backward
Fy{1} = [ zeros(1,is(2),is(3)); diff(F,1,1)];
Gy{1} = [ zeros(1,is(2),is(3)); diff(G,1,1)];
%forward
Fy{2} = [ diff(F,1,1); zeros(1,is(2),is(3))];
Gy{2} = [ diff(G,1,1); zeros(1,is(2),is(3))];
%central
FGy = 0.25*(Fy{1}+Fy{2}+Gy{1}+Gy{2});

Ft = F-G;


Mc = [ sum(FGy(:).^2), sum(FGy(:).*FGx(:)), sum(FGx(:).^2), ...
    sum(Ft(:).*FGy(:)), sum(Ft(:).*FGx(:))];

d = [Mc(1) Mc(2); Mc(2) Mc(3)]\[Mc(4); Mc(5)];

i = (d<0)+1;
M = [ sum(Fy{i(1)}(:).^2), sum(Fy{i(1)}(:).*Fx{i(2)}(:)), sum(Fx{i(2)}(:).^2), ... 
    sum(Ft(:).*Fy{i(1)}(:)), sum(Ft(:).*Fx{i(2)}(:))];

d = [M(1) M(2); M(2) M(3)]\[M(4); M(5)];

if nargout == 2
   figure; dispIm(sqrt((Fy{i(1)}*d(1)+Fx{i(2)}*d(2) - Ft).^2));
   e = sum((Fy{i(1)}(:)*d(1)+Fx{i(2)}(:)*d(2) - Ft(:)).^2).*trace(inv([M(1) M(2); M(2) M(3)]));
end