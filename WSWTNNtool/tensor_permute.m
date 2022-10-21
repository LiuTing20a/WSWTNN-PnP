function Out = tensor_permute(X,dim,k1,k2)
%%
if nargin < 4
    error('Not enough inputs!');
end
if k1 >= k2
    error('input k1 < k2');
end
if nargin == 4
    N = length(dim);
    if N == 2
        error('Please use the matrix transpose command!');
    else
        index =1:N;
        index([k1,k2])=[];
        tempOut = permute(X,[k1,k2,index]);
        Out = reshape(tempOut,dim(k1),dim(k2),[]);
    end
    
end
end



