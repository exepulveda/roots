function [Ip]=padded_image(I,scale)

[h, w, ~]=size(I);
Ip = zeros([ floor([h w]*(1+2*scale)) 3 ]);
%Ip(floor(scale*h) + (1:h), floor(scale*w) + (1:w),:) = I;
Ip(1:h, 1:w,:) = I;

end