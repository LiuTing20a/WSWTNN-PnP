% Combining Deep Denoiser and Low-rank Priors for Infrared Small Target Detection
% Corresponding Contributor: Ting Liu (Email: liuting@nudt.edu.cn)
% Author Afflications: National University of Defense Technology, China
% To run the code, you should install Matconvnet first.
%%The detailed code reproduction document can be downloaded from 
%%https://pan.baidu.com/s/1mPIl5XlYumnRwBC1zqudAA,and the key is 0640.
tic; 
clc;
clear;
close all;
addpath('functions/')                        % Add path
addpath('WSWTNNtool/')                       % Add path
saveDir= '..\WSWTNN-PnP公开\results\1\';     % save patch
imgpath='..\WSWTNN-PnP公开\data\1\';         % Data input path
imgDir = dir([imgpath '*.bmp']);             % List all files with the .bmp in the imgpath
%% patch parameters
patchSize =40;         % patch size
slideStep =40;         % sliding step
lambdaL = 0.8;         % L
len = length(imgDir);  % The length of imgDir
for i=1:len
    img = imread([imgpath imgDir(i).name]); % Read input data
    if ndims( img ) == 3
       img = rgb2gray(img);                 % Image graying
    end
    img = double(img);                      % Convert input data to double type
    [m n]=size(img);                        % The size of img
    %% constrcut patch tensor of original image
    tenD = gen_patch_ten(img, patchSize, slideStep);     % constrcut patch tensor 
    [n1,n2,n3] = size(tenD);                             % the size of patch tensor
    %% calculate prior weight map
    %      step 1: calculate two eigenvalues from structure tensor
    [lambda1, lambda2] = structure_tensor_lambda(img, 3);
    %      step 2: calculate corner strength function
    cornerStrength = (((lambda1.*lambda2)./(lambda1 + lambda2)));
    %      step 3: obtain final weight map
    maxValue = (max(lambda1,lambda2)); % Select the maximum of lambda1 and lambda2
    priorWeight = mat2gray(cornerStrength .* maxValue);
    %      step 4: constrcut patch tensor of weight map
    tenW = gen_patch_ten(priorWeight, patchSize, slideStep);
    %% The proposed WSWTNN-PnP model
    lambda4 = lambdaL / sqrt(max(n1,n2)*n3);            % regularization term
    [tenB,tenT] = trpca_WSWTNNpnp(tenD,lambda4,tenW);   % the proposed WSWTNN-PnP model
    %% recover the target and background image
    tarImg = res_patch_ten_mean(tenT, img, patchSize, slideStep);   % recover target image
    backImg = res_patch_ten_mean(tenB, img, patchSize, slideStep);  % recover background image
    maxv = max(max(double(img)));
    E = uint8(mat2gray(tarImg)*maxv);         % target image
    A = uint8(mat2gray(backImg)*maxv);        % background image
    %% save the results
    imwrite(E, [saveDir 'target/' imgDir(i).name]);     % Save target image 
    imwrite(A, [saveDir 'background/' imgDir(i).name]); % Save background image
end
toc;