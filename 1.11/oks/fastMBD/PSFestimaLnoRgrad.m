function [H, U, Report] = PSFestimaLnoRgrad(G, iH, PAR, Hstar)

% PSFestim
%
% Estimating PSFs but without the MC constraint matrix R
% solving u and h-step in FT
% in the h-step image gradient is used
% note: at this moment it works only for no SR srf=1, h-step is not
% properly written for srf>1
%%
% Using augmented Lagrangian approach
%
% gamma ... scalar; weight of the fidelity term
% mu ... scalar; weight of the blur consistency term
% lambda ... scalar; weight of the blur smoothing term (usually lambda=0)
% epsilon ... scalar; relaxation (only for TV) for numerical
% stability in the case |grad(U)| = 0

Report = [];

srf = PAR.srf;
if isfield(PAR,'spsf')
    spsf = PAR.spsf;
else
    if srf > 1
        [d is] = decmat(srf,[1 1],'o');
        spsf = full(unvec(d,is));
    else
        spsf = [1];
    end    
end
gamma = PAR.gamma;
lambda = PAR.lambda;
Lp = PAR.Lp;
reltol = PAR.reltol;
ccreltol = PAR.ccreltol;

% size of H
hsize = [size(iH,1) size(iH,2)];
% number of input images
P = size(G,3);
gsize = [size(G,1), size(G,2)];
usize = gsize*srf;
ssize = size(spsf);
hssize = hsize + ssize - 1;
% the block size that repeats in FFT
blsize = usize(1:2)/srf;
if PAR.verbose > 1
    FIG_HANDLE_H = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma,\lambda,\alpha,\beta = (',num2str([gamma,lambda,PAR.alpha_h,PAR.beta_h]),')']);
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

H = iH;

%% fft of sensor PSF
Fspsf = fft2(spsf,usize(1),usize(2));



%% Initialization of variables for min_U step, which do not change
% If we work with FFT, we have to move H center into the origin
%hshift = zeros(floor(hssize/2)+1); hshift(end) = 1;
% FU ... FFT of u
FU = fft2(U);
% FDx, FDx ... FFT of x and y derivative operators
FDx = fft2([1 -1],usize(1),usize(2));
FDy = fft2([1; -1],usize(1),usize(2));
DTD = conj(FDx).*FDx + conj(FDy).*FDy;

% FUx, FUy ... FFT of x (y) derivatives of U
FUx = zeros(usize);
FUy = zeros(usize);

eG = zeros(size(G));

% auxiliary variables for image gradient and blurs
% initialize to zeros
Vx = zeros(usize);
Vy = zeros(usize);
Vh = zeros([usize P]);
% extra variables for Bregman iterations
Bx = zeros(usize);
By = zeros(usize);
Bh = zeros([usize P]);

if doMSE
        Report.hstep.mse(1) = calculateMSE(H,Hstar);
end

for p = 1:P
   %%%eG(:,:,p) = edgetaper(G(:,:,p),conv2(H(:,:,p),spsf,'full'));
   eG(:,:,p) = edgetaper(G(:,:,p),ones(hsize)/prod(hsize));
   %eG(:,:,p) = G(:,:,p);
end
FeGu = repmat(fft2(eG),[srf srf 1]);
FeGx = repmat(FDx,[1 1 P]).*FeGu;
FeGy = repmat(FDy,[1 1 P]).*FeGu;
%FeGu = fft2(eG);

%% idea of Xu, Jia ECCV 2010
% determine usefulness of gradients
[M tau] = gradUsefulness(eG(:,:,1),hsize);
%tau

iterres_u = cell(PAR.maxiter_u,4);
iterres_h = cell(PAR.maxiter_h,4);

for mI = 1:PAR.maxiter
    
    Ustep;
    
    %% idea of Xu, Jia ECCV 2010
    % sharpen artifitially after every iteration the estimated image u
    %U = shock(U,5);
    %FU = fft2(U);
    %xD = real(ifft2(FDx.*FU));
    %yD = real(ifft2(FDy.*FU));
    
    %% idea of Xu, Jia ECCV 2010 
    % select only salient edges
    xD(M<tau) = 0; yD(M<tau) = 0;
    FUx = fft2(xD); FUy = fft2(yD);
    
    Hstep;
    if doMSE
        Report.hstep.mse(mI+1) = calculateMSE(H,Hstar);
    end
    gamma = gamma*1.3
    PAR.beta_h = PAR.beta_h*1.3;
    PAR.beta_u = PAR.beta_u*1.3;
    tau = tau/1.1;
end


%%% ***************************************************************
%%% min_U step
%%% ***************************************************************
function Ustep
% FHS ... FFT of  conv(H,spsf)
FHS = repmat(Fspsf,[1 1 P]).*fft2(H,usize(1),usize(2));

% FHTH ... sum_i conj(FHS_i)*FHS_i
FHTH = sum(conj(FHS).*FHS,3); 

% FGs ... FFT of H^T*D^T*g
% Note that we use edgetaper to reduce border effect
FGs = sum(conj(FHS).*FeGu,3);

beta = PAR.beta_u;
alpha = PAR.alpha_u;

%tic;
% main iteration loop, do everything in the FT domain
for i = 1:PAR.maxiter_u
    
    FUp = FU;
    b = FGs + beta/gamma*(conj(FDx).*fft2(Vx+Bx) + conj(FDy).*fft2(Vy+By));
    if srf > 1
        % CG solution
        [xmin,flag,relres,iter,resvec] = mycg(@gradcalcFU,vec(b),reltol,100,[],vec(FU));
        iterres_u(i,:) = {flag relres iter resvec};
        FU = unvec(xmin,usize);
        if PAR.verbose 
            disp(['beta, flag, iter:',num2str([beta flag iter])]);
        end
    else
    % or if srf == 1, we can find the solution in one step
        FU = b./( FHTH + beta/gamma*DTD);
        %disp(['beta: ', num2str(beta)]);
    end
    
    % Prepare my Lp prior
    Pr = asetupLnormPrior(Lp,alpha,beta);
    % get a new estimation of the auxiliary variable v
    % see eq. 2) in help above
    FUx = FDx.*FU;
    FUy = FDy.*FU;
    xD = real(ifft2(FUx));
    yD = real(ifft2(FUy));
    xDm = xD - Bx;
    yDm = yD - By;
    nDm = sqrt(xDm.^2 + yDm.^2);
    Vy = Pr.fh(yDm,nDm);
    Vx = Pr.fh(xDm,nDm);
    % update Bregman variables
    Bx = Bx + Vx - xD;
    By = By + Vy - yD;
    
    % we do not have to apply ifft after every iteration
    % this is only for convenience to display every new estimation
    %U = real((FU));
    %% impose constraints on U 
    %U = uConstr(U,vrange);
    E = sqrt(Vy.^2+Vx.^2);
    %updateFig(FIG_HANDLE,[],{i, [], []});
    updateFig(FIG_HANDLE_U,{[] By},{i, [], E},{[] Bx});
    % we increase beta after every iteration
    % it should help converegence but probably not necessary
    %beta = 2*beta;

    % Calculate relative convergence criterion
    relcon = sqrt(sum(abs(FUp(:)-FU(:)).^2))/sqrt(sum(abs(FU(:)).^2));
    
    if relcon < ccreltol
        break;
    end
end
if PAR.verbose
        disp(['min_U steps: ',num2str(i),' relcon:',num2str([relcon])]);
end
U = real(ifft2(FU));
%toc
%E = sqrt(Vy.^2+Vx.^2);
updateFig(FIG_HANDLE_U,[],{i, U, E});    

% the part of gradient calculated in every CG iteration
    function g = gradcalcFU(x)
        X = unvec(x,usize);
        g = 0;
        T = FHS.*repmat(X,[1 1 P]);
        for p=1:P
            % implementation of D^T*D in FT
            T(:,:,p) = repmat(reshape(sum(im2col(T(:,:,p),blsize,'distinct'),2)/(srf^2),blsize),[srf,srf]);
        end
        g = sum(conj(FHS).*T,3);
        g = vec(beta/gamma*DTD.*X + g);
    end
    
end
% end of Ustep

%%% ************************************************
%%% min_H step
%%% ************************************************
function Hstep

% FT of conv2(Ux,sensor_PSF) and conv2(Uy,sensor_PSF)
FUxS = repmat(FUx.*Fspsf,[1 1 P]);
FUyS = repmat(FUy.*Fspsf,[1 1 P]);
FUD = FeGx.*conj(FUxS) + FeGy.*conj(FUyS);
%FUD = FeGu.*repmat(conj(FU).*conj(Fspsf),[1 1 P]);

% here is the problem with SR, this can be used only if SRF=1
FUTU = conj(FUxS).*FUxS + conj(FUyS).*FUyS;
%FUTU = repmat(conj(FU).*FU,[1 1 P]);

beta = PAR.beta_h;
alpha = PAR.alpha_h;

FH = fft2(H,usize(1),usize(2));
%%%%
for i = 1:PAR.maxiter_h

FHp = FH; 
b = beta/gamma*fft2(Vh+Bh) + FUD;

if srf > 1
    % CG solution
    [xmin,flag,relres,iter,resvec] = mycg(@gradcalcFH,vec(b),reltol,100,[],vec(FH));
    iterres_h(i,:) = {flag relres iter resvec};
    FH = unvec(xmin,[usize P]);
    if PAR.verbose 
        disp(['beta, flag, iter:',num2str([beta flag iter])]);
    end
else
    FH = b./(FUTU + beta/gamma);
end

% Calculate relative convergence criterion
relcon = sqrt(sum(abs(FHp(:)-FH(:)).^2))/sqrt(sum(abs(FH(:)).^2));

%toc;

Pr = asetupLnormPrior(Lp,alpha,beta);     
hI = real(ifft2(FH));
hIm = hI - Bh;
nIm = abs(hIm);
Vh = Pr.fh(hIm,nIm);
%Vh = hIm;
Vh(Vh<0) = 0; % Forcing positivity this way is a correct approach!!!
%Vh(Vh<0.0003) = 0;
% force zero values on h ouside its support 
Vh(hsize(1)+1:end,:,:) = 0; Vh(1:hsize(1),hsize(2)+1:end,:) = 0;
% update Bregman variables
Bh = Bh + Vh - hI;

H = hI(1:hsize(1),1:hsize(2),:);

E = abs(Vh);

%updateFig(FIG_HANDLE, [], {[] [] reshape(E,hsize(1),hsize(2)*P)} , {i [reshape(convn(H,spsf,'full'),hssize(1),hssize(2)*P)]});
updateFig(FIG_HANDLE_H, {[] reshape(Bh,size(Bh,1),size(Bh,2)*P)}, ...
    {i, reshape(convn(H,spsf,'full'),hssize(1),hssize(2)*P), reshape(E,size(E,1),size(E,2)*P) },...
    []);


if relcon < ccreltol
    break;
end

end
if PAR.verbose
    %disp(['beta, flag, iter, relerr:',num2str([beta iterres{i,1} iterres{i,2} relcon])]);
    disp(['min_H step ',num2str(i),' relcon:',num2str([relcon])]);
end

    % the part of gradient calculated in every CG iteration
    function g = gradcalcFH(x)
        X = unvec(x,[usize P]);
        g = 0;
        Tx = FUxS.*X;
        Ty = FUyS.*X;
        for p=1:P
            % implementation of D^T*D in FT
            Tx(:,:,p) = repmat(reshape(sum(im2col(Tx(:,:,p),blsize,'distinct'),2)/(srf^2),blsize),[srf,srf]);
            Ty(:,:,p) = repmat(reshape(sum(im2col(Ty(:,:,p),blsize,'distinct'),2)/(srf^2),blsize),[srf,srf]);
        end
        g = conj(FUxS).*Tx + conj(FUyS).*Ty;
        g = vec(beta/gamma*X + g);
    end


    function [y,g,H] = minHcon(x)
        Ax = A*x; %Afun(x);
        %y = 0.5*(x'*Ax - 2*x'*b + zTz);
        y = 0.5*sum((Ax-b).^2);
        if nargout > 1
          g = Ax - b;
          if nargout > 2
              H = A;
          end
        end
    end

    function y = Afun(x)
        y = A*x;
    end
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


  

