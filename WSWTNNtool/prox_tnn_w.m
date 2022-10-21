function [X, tnn, trank] = prox_tnn_w(Y, rho)

% The proximal operator of the tensor nuclear norm of a 3-order tensor
%
% min_X rho*||X||_*+0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
%
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X

dim = ndims(Y);
[n1, n2, n3] = size(Y);
n12 = min(n1, n2);
Yf = fft(Y, [], dim);
Uf = zeros(n1, n12, n3);
Vf = zeros(n2, n12, n3);
Sf = zeros(n12,n12, n3);

Yf(isnan(Yf)) = 0;
Yf(isinf(Yf)) = 0;

trank = 0;
endValue = int16(n3/2 + 1);
for i = 1 : endValue
    [Uf(:,:,i), Sf(:,:,i), Vf(:,:,i)] = svd(Yf(:,:,i), 'econ');
    s1 = diag(Sf(:, :, i));
%     s = max(s - rho, 0);
         sikema = 10^-6;%sikema = 10^-6;
    diagsj_w = 1./(s1+sikema);%奇异值的反比例系数
    diagsj_w = diagsj_w/(diagsj_w(7));%使得第rank个权值是1，后面的权值，应该是递增的，前面的则是递减
    %可以近似保留前面的rank个秩，同时对后面的，也能进行一定的削减
    s=max(s1-rho*diagsj_w,0);
    Sf(:, :, i) = diag(s);
    temp = length(find(s>0));
    trank = max(temp, trank);
end
for j =n3:-1:endValue+1
    Uf(:,:,j) = conj(Uf(:,:,n3-j+2));
    Vf(:,:,j) = conj(Vf(:,:,n3-j+2));
    Sf(:,:,j) = Sf(:,:,n3-j+2);
end

Uf = Uf(:, 1:trank, :);
Vf = Vf(:, 1:trank, :);
Sf = Sf(1:trank, 1:trank, :);

U = ifft(Uf, [], dim);
S = ifft(Sf, [], dim);
V = ifft(Vf, [], dim);

X = tprod( tprod(U,S), tran(V) );
tnn = sum( diag( Sf(:,:,1) ) );
end