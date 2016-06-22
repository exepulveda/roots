function [I m v] = simpnormimg(G)

% Normalize images in G
%
% I = normimg(G)
%
% G ... cell array; input images
%
% normalize the input images so that intensity values lie between 0 and 1.
%

I = cell(size(G));

lb = min(vec([G{:}]));
ub = max(vec([G{:}]));

v = ub-lb;
m = lb;
for i=1:numel(G)
    I{i} = (double(G{i})-m)/v;
end



