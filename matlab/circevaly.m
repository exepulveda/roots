function x = circevaly(x, circ, imsize)

w = imsize(2);

a = sqrt( circ.r^2 - (x - circ.c(2)).^2 );

x1 = circ.c(1) + a;
v1 = sum(x1>=0 & x1< w);
    
x2 = circ.c(1) - a;
v2 = sum(x2>=0 & x2< w);


if v1>v2
    x = x1;
else
    x = x2;
end