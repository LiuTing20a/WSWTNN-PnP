function [B,T] = trpca_WSWTNNpnp(X, lambda, tenW, opts)
%% Parameter Settings
N1 = rankN(X,0.1);
[w, h, c] = size(X);   % The size of the X
Nway = size(X);        % The size of the X
Ndim = ndims(X);       % The number of dimensions of X
alpha=[0,  0.001,  1;  % alpha value
       0,    0,    1;
       0,    0,    0];
mu = 2*1e-3;                                % Regularization parameter
opts.alpha=alpha/sum(alpha(:));             % alpha value 
opts.tol = 1e-4;                            % Iteration termination parameter
opts.max_iter = 500;                        % Number of Iteration
opts.gamma = 1.2;                           % gamma value
omega =1000;                                % Threshold in t-SVT (¦Ä)
opts.beta = opts.alpha/omega;               % beta value
opts.rho = 1/omega;                         % rho value
opts.max_beta = 1e10*ones(Ndim,Ndim);       % Maximum value of beta
opts.max_rho = 1e10;                        % Maximum value of rho
DEBUG = 1;
if ~exist('opts', 'var')
    opts = [];
end
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'max_rho');      max_rho = opts.max_rho;        end
if isfield(opts, 'beta');        beta = opts.beta;                end
if isfield(opts, 'max_beta');      max_beta = opts.max_beta;        end
if isfield(opts, 'alpha');        alpha = opts.alpha;                end
if isfield(opts, 'gamma');         gamma = opts.gamma;              end
if isfield(opts, 'lambda');        lambda = opts.lambda;                end
if isfield(opts, 'N1');           N1 = opts.N1;                  end
if isfield(opts, 'mu');          mu = opts.mu;                end
%% FFDnet parameter
sigma = 1;                              % sigma value                              
useGPU      = 1;                        % GPU           
load(fullfile('FFDNet_Clip_gray.mat')); % Load the trained FFDNet 
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu');
end
%% Initialization
weightTen = ones(Nway);  % Initialization weight matrix
N = ndims(X);            % The number of dimensions of X
B = zeros(Nway);         % Initialization background patch-tensor B
T = zeros(Nway);         % Initialization target patch-tensor T
Z = cell(N,N);           % Initialization auxiliary tensor Z
for i=1:N
    for j=1:N
        Z{i,j} = zeros(Nway); % the auxiliary tensor Z
    end
end
L=B;                          % the auxiliary tensor L
P = Z;                        % multiplier P
M = zeros(Nway);              % multiplier M
M1 = zeros(Nway);             % multiplier M1
temp = Z;
for iter = 1 : max_iter
    preT = sum(T(:) > 0);
    %% Let
    Lold = B;
    %% Update Z with other variables being fixed:
    tau = alpha./beta;        % threshold tau
    for i=1:N-1
        for j=i+1:N
            % B is expanded by the mode-ij,and Nway is the order of B.
            tempL   = tensor_permute(B,Nway,i,j);
            % P{i,j} is expanded by the mode-ij,and Nway is the order of P{i,j}.
            tempP   = tensor_permute(P{i,j},Nway,i,j);
            % Updating Z correspond to Eq.(11) in the proposed WSWTNN-PnP method
            Z{i,j}  = tensor_ipermute(prox_tnn_w(tempL+tempP/beta(i,j),tau(i,j)),Nway,i,j);
            temp{i,j} = Z{i,j}-P{i,j}/beta(i,j);
        end
    end
    %% Update B with other variables being fixed
    tempsum = zeros(Nway);
    for i=1:N-1
        for j=1+i:N
            tempsum = tempsum+ beta(i,j)*temp{i,j};
        end
    end
    % Updating B correspond to Eq.(13) in the proposed WSWTNN-PnP method
    B = (tempsum+rho*(X-T+M/rho)+rho*(L-M1/rho))/(2*rho+sum(beta(:)));
    %% Update L with other variables being fixed:
    input = unorigami(B+M1/rho,[w h c]);    % The input of the deep denoiser
    sigX = unorigami(B,[w h c]);
    input = single(input);
    if c==3
        if mod(w,2)==1
            input = cat(1,input, input(end,:,:)) ;
        end
        if mod(h,2)==1
            input = cat(2,input, input(:,end,:)) ;
        end
    else
        if mod(w,2)==1
            input = cat(1,input, input(end,:)) ;
        end
        if mod(h,2)==1
            input = cat(2,input, input(:,end)) ;
        end
    end
    
    if useGPU
        input = gpuArray(input);
    end
    max_in = max(input(:));min_in = min(input(:));
    input = (input-min_in)/(max_in-min_in);
    sigmas = sigma/(max_in-min_in);
    % perform denoising
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
    output = res(end).x;
    output(output<0)=0;output(output>1)=1;
    output = output*(max_in-min_in)+min_in;
    
    if c==3
        if mod(w,2)==1
            output = output(1:end-1,:,:);
        end
        if mod(h,2)==1
            output = output(:,1:end-1,:);
        end
    else
        if mod(w,2)==1
            output = output(1:end-1,:);
        end
        if mod(h,2)==1
            output = output(:,1:end-1);
        end
    end
    
    if useGPU
        output = gather(output);
    end
    if c==3
        L = double(output);
    else
    % Updating L correspond to Eq.(16) in the proposed WSWTNN-PnP method
        L = origami(double(output),[w h c]);    
    end
    %% Update T with other variables being fixed:
    % Updating T correspond to Eq.(18) in the proposed WSWTNN-PnP method
    T = prox_l1(X-B+M/rho,weightTen*lambda/rho); 
    % Calculating local weight map W
    weightTen = N1./ (abs(T) + 0.01)./tenW;    
    %% check the convergence
    dM = X-B-T;
    err = norm(dM(:))/norm(X(:));                
    if DEBUG
        if iter == 1 || mod(iter, 1) == 0
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                ', err=' num2str(err)...
                ',|T|0 = ' num2str(sum(T(:) > 0))]);
        end
    end
    currT = sum(T(:) > 0);
    if err < tol || (preT>0 && currT>0 && preT == currT) % Convergence condition
        break;
    end
    %% Update multipliers
    for i=1:N-1
        for j=i:N
    % Updating multipliers P{i,j} correspond to Eq.(19) in the proposed WSWTNN-PnP method
            P{i,j} = P{i,j}+beta(i,j) * (B-Z{i,j}); 
        end
    end
    % Updating multiplier M correspond to Eq.(19) in the proposed WSWTNN-PnP method
    M = M + rho*dM;   
    % Updating multiplier M1 correspond to Eq.(19) in the proposed WSWTNN-PnP method
    M1 = M1 + rho*(B-L);                
    beta = min(gamma*beta,max_beta);    % updating beta
    rho = min(gamma*rho,max_rho);       % updating rho
end

function N = rankN(X, ratioN)
    [~,~,n3] = size(X);
    D = Unfold(X,n3,1);
    [~, S, ~] = svd(D, 'econ');
    [desS, ~] = sort(diag(S), 'descend');
    ratioVec = desS / desS(1);
    idxArr = find(ratioVec < ratioN);
    if idxArr(1) > 1
        N = idxArr(1) - 1;
    else
        N = 1;
    end