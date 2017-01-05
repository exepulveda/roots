function p = findcorner(circ1, circ2, ref)
% Find a corner p defined by the intersection of two cirles 
% ref  -   reference point


R = circ1.r;
d = norm(circ2.c-circ1.c);
r = circ2.r;

ang = circinter(R, d, r); 

pc = R*((circ2.c-circ1.c)/d);
p1 = circ1.c + ([cos(ang) -sin(ang); sin(ang) cos(ang)]*pc')';
p2 = circ1.c + ([cos(-ang) -sin(-ang); sin(-ang) cos(-ang)]*pc')';


d1 = norm(p1-ref);
d2 = norm(p2-ref);

if d1<d2
    p = p1;
else
    p = p2;
end

% if insideImage(p, imsize)
%     return
% end
return


% function resp = insideImage(p, imsize)
% x = p(1);
% y = p(2);
% 
% h = imsize(1);
% w = imsize(2);
% 
% if x<0 || x> w || y<0 || y>h
%     resp = false;
% else
%     resp = true;
% end
% return