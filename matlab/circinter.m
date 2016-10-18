function angle = circinter(R, d, r)
        
    if (d<=eps)
        angle = pi;
        return
    end

% //    if( fabs(d-(R+r))<eps)
% //    {
% //        return 0;
% //    }

    x = (d*d-r*r+R*R)/(2*d);

    rat = x/R;
    if (rat<=-1.0)
        angle= pi;
        return
    end

    angle= acos(rat);

return