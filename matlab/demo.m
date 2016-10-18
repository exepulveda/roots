
clear all
close all

addpath 1.11
addpath CV/CV/CameraGeometry

imageName='1.11-776.jpg';

I_original = imread(imageName);


%% Prepare input
padding = 3;

I_pad = I_original(padding +1:end-padding, padding +1:end-padding,:);

I= rgb2gray(I_pad);

I = imgaussfilt(I,1); % gauss filter
figure;imshow(I)

se = strel('square',7);
dilatedI= imdilate(I,se);

figure, imshow(dilatedI)


se = strel('square',3);
erodedI= imerode(dilatedI,se);
figure, imshow(erodedI)


%% Edge filter

I = edge(erodedI,'Canny', .3); 
figure;imshow(I)

%% Detect lines
[H,theta,rho] = hough(I);

% %display transform
% figure
% imshow(imadjust(mat2gray(H)),[],...
%        'XData',theta,...
%        'YData',rho,...
%        'InitialMagnification','fit');
% xlabel('\theta (degrees)')
% ylabel('\rho')
% axis on
% axis normal
% hold on
% colormap(hot)
% figure;
 P = houghpeaks(H,15,'threshold',ceil(0.3*max(H(:))));
 x = theta(P(:,2));
 y = rho(P(:,1));
% plot(x,y,'s','color','black');


lines = houghlines(I,theta,rho,P,'FillGap',5,'MinLength',10);

figure, imshow(I), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

end

%%Detect corners:

%Find corners iterating over all segments. 
%y
%^ A  B  C
%  H     D  
%| G  F  E
%|---->x


[xx,yy] = ginput(8);

Pts = [xx yy];

h=size(I,1);
w=size(I,2);

optPts = [
    0   0
    w/2 0
    w   0
    w   h/2
    w   h
    w/2   h
    0 h
    0   h/2
    ]



% for k = 1:length(lines)
%    
%    a=lines(k).point1; 
%    b=lines(k).point2;
%    
%    d=min(norm(Aopt-a), norm(Aopt-b));
%    if d<dA
%        dA = d;
%        if norm(Aopt-a)< norm(Aopt-b)
%            A=a;
%        else
%            A=b;
%        end
%    end
%    
%     d=min(norm(Bopt-a), norm(Bopt-b));
%    if d<dB
%        dB = d;
%        if norm(Bopt-a)< norm(Bopt-b)
%            B=a;
%        else
%            B=b;
%        end
%    end
%    
%     d=min(norm(Copt-a), norm(Copt-b));
%    if d<dC
%        dC = d;
%        if norm(Copt-a)< norm(Copt-b)
%            C=a;
%        else
%            C=b;
%        end
%    end
%    
%    
%     d=min(norm(Dopt-a), norm(Dopt-b));
%    if d<dA
%        dD = d;
%        if norm(Dopt-a)< norm(Dopt-b)
%            D=a;
%        else
%            D=b;
%        end
%    end
%    
% end

% plot(w/2,h-26,'*m');
% plot(.25*w,h-17,'*m');
% plot(.67*w,h-20,'*m');

% 
% plot (A(1), A(2),'*m')
% plot (B(1), B(2),'*m')
% plot (C(1), C(2),'*m')
% plot (D(1), D(2),'*m')

for i=1:size(Pts,1)
    plot(Pts(i,1),Pts(i,2), '*m')
    hold on
end



%% Obtain H

p2=Pts';
p1=optPts';


H = DirectLinearTransformation(p1,p2);




 
%% Plot result

I2p = padded_image(I_pad,.02);

imgwarped = imagewarping(I2p,I2p,H);

%   imgblended =imageBlending(I1p,imgwarped);

figure;
subplot(1,2,1)
imshow(I_original);
subplot(1,2,2)
imshow(imgwarped);

 