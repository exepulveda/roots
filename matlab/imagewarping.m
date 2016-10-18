function [ imgwarped ] = imagewarping(img1,img2,H12)

imgg2 = rgb2gray(uint8(img2));


[height1,width1,~] = size(img1);
[height2,width2,~] = size(img2);

imgwarped = zeros(size(img1));

%     figure(1); clf; imshow([img1 img2]); hold on;
%     set(figure(1),'outerposition',[-1800 350 1420 620]);
%     plot(pts1(1,:),pts1(2,:),'b.','markersize',8);
%     plot(pts2(1,:)+width,pts2(2,:),'b.','markersize',8);

[~,~,d] = size(img1);
[posx,posy] = meshgrid(1:width1,1:height1);
pos1 = [posx(:)';posy(:)'];
pos2 = H12 * makehomogeneous(pos1);
pos2 = round(makeinhomogeneous(pos2));
in = inpolygon(pos2(1,:),pos2(2,:),[1 width1 width1 1 1],[1 1 height1 height1 1]);
pos1 = pos1(:,in);
pos2 = pos2(:,in);

in = inpolygon(pos2(1,:),pos2(2,:),[1 width2 width2 1 1],[1 1 height2 height2 1]);
pos1 = pos1(:,in);
pos2 = pos2(:,in);


img2 = double(img2);
[y,x] = find(imgg2 == 0);

if d==3
    for i = 1:length(x)
        img2(y(i),x(i),1) = 0;
        img2(y(i),x(i),2) = 0;
        img2(y(i),x(i),3) = 0;
    end
elseif d==1
    for i = 1:length(x)
        img2(y(i),x(i)) = 0;
        img2(y(i),x(i)) = 0;
        img2(y(i),x(i)) = 0;
    end
end

%     plot(x+width,y,'b.');
%     intsetx = intersect(pos2(1,:),x(:));
%     intsety = intersect(pos2(2,:),y(:));
%
%     ind = intersect(intsetx,intsety);
%     pos2(:,ind) = [];
%
%     plot(pos1(1,:),pos1(2,:),'b.');
%     plot(pos2(1,:)+width,pos2(2,:),'b.');

if d==3
    for i = 1:size(pos1,2);
        %         plot(1:width,i,'r.');
        imgwarped(pos1(2,i),pos1(1,i),1) = img2(pos2(2,i),pos2(1,i),1);
        imgwarped(pos1(2,i),pos1(1,i),2) = img2(pos2(2,i),pos2(1,i),2);
        imgwarped(pos1(2,i),pos1(1,i),3) = img2(pos2(2,i),pos2(1,i),3);
    end
elseif d==1
    for i = 1:size(pos1,2);
        %         plot(1:width,i,'r.');
        imgwarped(pos1(2,i),pos1(1,i)) = img2(pos2(2,i),pos2(1,i));
        imgwarped(pos1(2,i),pos1(1,i)) = img2(pos2(2,i),pos2(1,i));
        imgwarped(pos1(2,i),pos1(1,i)) = img2(pos2(2,i),pos2(1,i));
    end
end
% figure; imshow(uint8(imgwarped));
if d == 3
    imgwarped = uint8(imgwarped);
end

end
