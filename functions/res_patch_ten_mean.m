function recImg = res_patch_ten_mean(patchTen, img, patchSize, slideStep)

% 2017-07-31
% This matlab code implements the RIPT model for infrared target-background 
% separation.
%
% Yimian Dai. Questions? yimian.dai@gmail.com
% Copyright: College of Electronic and Information Engineering, 
%            Nanjing University of Aeronautics and Astronautics
%[n1,n2,n3] = size(tenD); %
[imgHei, imgWid] = size(img);

rowPatchNum = ceil((imgHei - patchSize) / slideStep) + 1;
colPatchNum = ceil((imgWid - patchSize) / slideStep) + 1;
rowPosArr = [1 : slideStep : (rowPatchNum - 1) * slideStep, imgHei - patchSize + 1];
colPosArr = [1 : slideStep : (colPatchNum - 1) * slideStep, imgWid - patchSize + 1];

%% for-loop version
accImg = zeros(imgHei, imgWid);
weiImg = zeros(imgHei, imgWid);
k = 0;
onesMat = ones(patchSize, patchSize);
for col = colPosArr
    for row = rowPosArr
        k = k + 1;
       % [a,b]= size(patchTen(:, :, k));
        [a1,b1]= size(patchSize);
        [a2,b2]= size(patchSize);
        %B1= imresize(patchTen(:, :, k),[patchSize patchSize]);%
        tmpPatch = reshape(patchTen(:, :, k), [patchSize, patchSize]);
        %tmpPatch = reshape(B1, [patchSize, patchSize]);
        %tmpPatch = reshape(patchTen(:, :, k), [49, 1600]);
        %tmpPatch = reshape(patchTen(:, :, k), [n3, n1*n2]);
        accImg(row : row + patchSize - 1, col : col + patchSize - 1) = tmpPatch;
        weiImg(row : row + patchSize - 1, col : col + patchSize - 1) = onesMat;
    end
end

recImg = accImg ./ weiImg;
