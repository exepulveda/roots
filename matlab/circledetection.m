function [centre, rad, base] = circledetection(im)

[h, w] = size(im);

if h>w %vertical sampling 
    verticalSampling = 1;
    minrad = 1.5*h;
else % horisontal sampling
    verticalSampling = 0;
    minrad = w;
end

if verticalSampling % W E
    y = [h/6; h/2; 5*h/6];
    mu = (w/2)*[1 1 1]';
    sigma = w;
else % N S
    x = [w/6; w/2; 5*w/6];
    mu = (h/2)*[1 1 1]';
    sigma = h;
end


% plot_circle(centre, rad);
best_q = 0; %quality(im, centre, rad);

for i=1:800
    rad = 0;
    while rad<minrad
        if verticalSampling % W E
            if i<=400
                x = w*rand(3,1);
            else 
                x = mu + randn(3,1)*chol(sigma);
                sigma = .99*sigma;
            end
        else % N S
            if i<=400 
                y = h*rand(3,1);
            else
                y = mu + randn(3,1)*chol(sigma);
                sigma = .99*sigma;
            end
        end
        
        base = [x y];
        [centre, rad] = calcCircle( base(1,:), base(2,:), base(3,:));
    end
    
    % plot_circle(centre, rad);
    q = quality(im, centre, rad);
    if q>best_q
       best_q = q; 
       best_centre = centre;
       best_rad = rad;
       best_base = base;
       
       if verticalSampling % W E
           mu = x;
       else % N S
           mu = y;
       end

    end

end

centre = best_centre;
rad = best_rad;
base = best_base;

return


function q = quality(im, centre, r) 
assert(r>0);

[h,w]=size(im);

if h>w  % W E
    yp = 1:h;
    xp = centre(1) + sqrt( r^2 - (yp - centre(2)).^2  );
    
    val1 = improfile(im,xp,yp);
    val1 = val1(~isnan(val1));
    val1 = val1.^2;
    val1 = sum(val1);
    
    xp = centre(1) - sqrt( r^2 - (yp - centre(2)).^2  );
    val2 = improfile(im,xp,yp);
    val2 = val2(~isnan(val2));
    val2 = val2.^2;
    val2 = sum(val2);
    
else  % N S
    xp = 1:w;
    yp = centre(2) + sqrt( r^2 - (xp - centre(1)).^2  );
    
    val1 = improfile(im,xp,yp);
    val1 = val1(~isnan(val1));
    val1 = val1.^2;
    val1 = sum(val1);
    
    yp = centre(2) - sqrt( r^2 - (xp - centre(1)).^2  );
    val2 = improfile(im,xp,yp);
    val2 = val2(~isnan(val2));
    val2 = val2.^2;
    val2 = sum(val2);
end

q = max(val1, val2);

return