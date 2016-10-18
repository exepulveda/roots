function y = circevalx(x, circ, imsize)

h = imsize(1);

a = sqrt( circ.r^2 - (x - circ.c(1)).^2 );

y1 = circ.c(2) + a;
v1 = sum(y1>=0 & y1< h);
    
y2 = circ.c(2) - a;
v2 = sum(y2>=0 & y2< h);


if v1>v2
    y = y1;
else
    y = y2;
end