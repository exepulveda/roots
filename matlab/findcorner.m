function p = findcorner(circ1, circ2, imsize)
% Find a corner p defined by the intersection of two cirles 


R = circ1.r;
d = norm(circ2.c-circ1.c);
r = circ2.r;

ang = circinter(R, d, r); 

pc = R*((circ2.c-circ1.c)/d);
p = circ1.c + ([cos(ang) -sin(ang); sin(ang) cos(ang)]*pc')';

if insideImage(p, imsize)
    return
end

p = circ1.c + ([cos(-ang) -sin(-ang); sin(-ang) cos(-ang)]*pc')';

if insideImage(p, imsize)
    return
end

p = [-1, -1];

return

function resp = insideImage(p, imsize)
x = p(1);
y = p(2);

h = imsize(1);
w = imsize(2);

if x<0 || x> w || y<0 || y>h
    resp = false;
else
    resp = true;
end
return