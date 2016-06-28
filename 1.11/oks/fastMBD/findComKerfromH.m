function [H, U, Report] = findComKerfromH(G, iU, Hstar)

% find Common Kernel from PSFs in G
%
%
% note: trying to solve h-step in FT
% at this moment it works only for no SR srf=1
%
% Solving linear problem Ax=b that is (gamma*U'U  + delta*R)h = gamma*U'g
% using Gaussian elimination method
% or using fmincon to impose the contraint that the int. values 
% of PSFs must lie between 0 and 1. Fmincon is necessary if PSFs 
% are overestimated. 
%
% Using augmented Lagrangian approach
%
% note: If G is a cell array, output H will be a cell array as well

Report = [];

PAR.verbose = 0; %{0 = no messages,1 = text messages,2 = text and graphs}

% common parameters to both PSF and image estimation
% data term weight gamma
PAR.gamma = 1e1;
% which Lp norm to use
PAR.Lp = 1;

% PSFs estimation
PAR.beta_h = 1e1*PAR.gamma;
PAR.alpha_h = 1;

% image estimation
% 1e3*PAR.gamma
PAR.beta_u = 1e1*PAR.gamma;
PAR.alpha_u = 1;
% number of iterations
PAR.maxiter_u = 100;
PAR.maxiter_h = 100;
PAR.maxiter = 100;
PAR.ccreltol = 1e-4;

gamma = PAR.gamma;
Lp = 1;
ccreltol = PAR.ccreltol;

% conver cell input to 3D matrix
if iscell(G)
    inputIsCell = 1;
    G = reshape([G{:}],size(G{1},1),size(G{1},2),length(G));
else
    inputIsCell = 0;
end
% number of input images
P = size(G,3);
gsize = [size(G,1), size(G,2)];
usize = gsize;

hsize = gsize;
cen = (hsize+1)/2;
iH = zeros([hsize,P]);
for j=1:P
    iH(:,:,j) = initblur(hsize,cen,[1 1]);
end
% the block size that repeats in FFT
if PAR.verbose > 1
    FIG_HANDLE_H = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma, \alpha,\beta = (',num2str([gamma,PAR.alpha_h,PAR.beta_h]),')']);
    FIG_HANDLE_U = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma,\alpha,\beta = (',num2str([gamma,PAR.alpha_u,PAR.beta_u]),')']);
else
    FIG_HANDLE_H = [];
    FIG_HANDLE_U = [];
end

% if true PSF Hstar is provided -> calculate MSE
if exist('Hstar','var') && ~isempty(Hstar)
    doMSE = 1;
    Report.hstep.mse =  zeros(1,PAR.maxiter+1);
else
    doMSE = 0;
end

U = zeros(usize);
%U = iU;
U(1) = 1;

H = iH;

%% Initialization of variables for min_U step, which do not change
% If we work with FFT, we have to move H center into the origin
%hshift = zeros(floor(usize/2)+1); hshift(end) = 1;
hshift = 1;
%hshift(1) = 1;
% FU ... FFT of u
FU = fft2(U);
% FDx, FDx ... FFT of x and y derivative operators
FDx = fft2([1 -1],usize(1),usize(2));
FDy = fft2([1; -1],usize(1),usize(2));
DTD = conj(FDx).*FDx + conj(FDy).*FDy;

eG = zeros(size(G));
%eG = zeros([srf srf 1].*size(G));

% auxiliary variables for image gradient and blurs
% initialize to zeros
Vx = zeros(usize);
Vy = zeros(usize);
Vu = zeros(usize);
Vh = zeros([usize P]);
% extra variables for Bregman iterations
Bx = zeros(usize);
By = zeros(usize);
Bu = zeros(usize);
Bh = zeros([usize P]);

if doMSE
        Report.hstep.mse(1) = calculateMSE(H,Hstar);
end
for p = 1:P
        %eG(:,:,p) = edgetaper(real(ifft2(repmat(fft2(G(:,:,p)),[srf srf]))),conv2(H(:,:,p),spsf,'full'));
        %eG(:,:,p) = edgetaper(G(:,:,p),conv2(H(:,:,p),spsf,'full'));
        eG(:,:,p) = G(:,:,p);
    end
FeGu = fft2(eG);
for mI = 1:PAR.maxiter
    
%    Ustep;
    %U = shock(U,20);
    %FU = fft2(U);
    Hstep;
    Ustep;
    if doMSE
        Report.hstep.mse(mI+1) = calculateMSE(H,Hstar);
    end
end

if inputIsCell
    H = reshape(mat2cell(H,size(H,1),size(H,2),ones(1,size(H,3))),1,[]);
end
%% Initialization of variables for min_H step, which depend on U 


%%% ***************************************************************
%%% min_U step
%%% ***************************************************************
function Ustep
% FH ... FFT of  H
FH = repmat(conj(fft2(hshift,usize(1),usize(2))),[1 1 P])...
        .*fft2(H,usize(1),usize(2));

% FHTH ... sum_i conj(FH_i)*FH_i
FHTH = sum(conj(FH).*FH,3); 

% FGs ... FFT of H^T*D^T*g
% Note that we use edgetaper to reduce border effect
%FGs = zeros(usize);
%FGu = zeros(usize);
FGs = sum(conj(FH).*FeGu,3);

%for p=1:P
%    %eG = edgetaper(G(:,:,p),H(:,:,p));
%    eG = G(:,:,p);
%    FGu = repmat(fft2(eG),[srf srf 1]);
%    FGs = FGs + conj(FH(:,:,p)).*FGu;
%end
beta = PAR.beta_u;
alpha = PAR.alpha_u;

% main iteration loop, do everything in the FT domain
for i = 1:PAR.maxiter_u
    
    FUp = FU;
    %b = FGs + beta/gamma*(conj(FDx).*fft2(Vx+Bx) + conj(FDy).*fft2(Vy+By));
    b = FGs + beta/gamma*fft2(Vu+Bu);
    %FU = b./( FHTH + beta/gamma*DTD);
    FU = b./( FHTH + beta/gamma);
    
    % Prepare my Lp prior
    %Pr = asetupLnormPrior(Lp,alpha,beta);
    % get a new estimation of the auxiliary variable v
    % see eq. 2) in help above 
    uD = real(ifft2(FU));
    %xD = real(ifft2(FDx.*FU));
    %yD = real(ifft2(FDy.*FU));
    uDm = uD - Bu;
    %xDm = xD - Bx;
    %yDm = yD - By;
    %nDm = sqrt(xDm.^2 + yDm.^2);
    %Vu = Pr.fh(uDm,abs(uDm));
    Vu = uDm;
    Vu(Vu<0) = 0;
    %Vy = Pr.fh(yDm,nDm);
    %Vx = Pr.fh(xDm,nDm);
    
    % update Bregman variables
    Bu = Bu + Vu - uD;
    %Bx = Bx + Vx - xD;
    %By = By + Vy - yD;
    
    % we do not have to apply ifft after every iteration
    % this is only for convenience to display every new estimation
    %U = real((FU));
    %% impose constraints on U 
    %U = uConstr(U,vrange);
    %E = sqrt(Vy.^2+Vx.^2);
    E = abs(Vu);
    %updateFig(FIG_HANDLE,[],{i, [], []});
    updateFig(FIG_HANDLE_U,{[] Bu},{i, [], E});
    % we increase beta after every iteration
    % it should help converegence but probably not necessary
    %beta = 2*beta;

    % Calculate relative convergence criterion
    relcon = sqrt(sum(abs(FUp(:)-FU(:)).^2))/sqrt(sum(abs(FU(:)).^2));
    if PAR.verbose
        disp(['relcon:',num2str([relcon])]);
    end
    if relcon < ccreltol
        break;
    end
end
disp(['min_U steps: ',num2str(i)]);
U = real(ifft2(FU));
%E = sqrt(Vy.^2+Vx.^2);
updateFig(FIG_HANDLE_U,[],{i, U, E});    


    
end
% end of Ustep

%%% ************************************************
%%% min_H step
%%% ************************************************
function Hstep


FUD = FeGu.*repmat(conj(FU),[1 1 P]);

FUTU = repmat(conj(FU).*FU,[1 1 P]);


iterres = cell(PAR.maxiter_h,2);
beta = PAR.beta_h;
alpha = PAR.alpha_h;

FH = fft2(H,usize(1),usize(2));
%%%%
for i = 1:PAR.maxiter_h

FHp = FH; 
b = beta/gamma*fft2(Vh+Bh) + FUD;
FH = b./(FUTU + beta/gamma);



%flag
%%%%%
% Calculate relative convergence criterion
relcon = sqrt(sum(abs(FHp(:)-FH(:)).^2))/sqrt(sum(abs(FH(:)).^2));

%toc;
%H = unvec(xmin,size(H));

%Pr = asetupLnormPrior(Lp,alpha,beta);     
hI = real(ifft2(FH));
hIm = hI - Bh;
nIm = abs(hIm);
%Vh = Pr.fh(hIm,nIm);
Vh = hIm;
Vh(Vh<0) = 0; % Forcing positivity this way is a correct approach!!!
% force zero values on h ouside its support 
%Vh(hsize(1)+1:end,:,:) = 0; Vh(1:hsize(1),hsize(2)+1:end,:) = 0;
% update Bregman variables
Bh = Bh + Vh - hI;

H = hI(1:hsize(1),1:hsize(2),:);

E = abs(Vh);

%updateFig(FIG_HANDLE, [], {[] [] reshape(E,hsize(1),hsize(2)*P)} , {i [reshape(convn(H,spsf,'full'),hssize(1),hssize(2)*P)]});
updateFig(FIG_HANDLE_H, {[] reshape(Bh,size(Bh,1),size(Bh,2)*P)}, ...
    {i, reshape(H,hsize(1),hsize(2)*P), reshape(E,size(E,1),size(E,2)*P) },...
    []);

%beta = 2*beta;

if PAR.verbose
    %disp(['beta, flag, iter, relerr:',num2str([beta iterres{i,1} iterres{i,2} relcon])]);
    disp(['relcon:',num2str([relcon])]);
end
if relcon < ccreltol
    break;
end

end
disp(['min_H steps: ',num2str(i)]);

end
% end of Hstep

end

function r = calculateMSE(h,hs)
    hsize = size(hs);
    i = size(h)-hsize+1;
    R = zeros(prod(hsize(1:2)),prod(i(1:2)),hsize(3));
    h = h/sum(h(:))*sum(hs(:));
    for p = 1:hsize(3)
        R(:,:,p) = im2col(h(:,:,p),[size(hs,1) size(hs,2)],'sliding');
    end
    %nhs = norm(hs(:));
    s = sqrt(sum(sum((R-repmat(reshape(hs,prod(hsize(1:2)),1,hsize(3)),1,prod(i(1:2)))).^2,3),1));
    %r = min(s);
    r = s(ceil(prod(i(1:2))/2));
end


  

