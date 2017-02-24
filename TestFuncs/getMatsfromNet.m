function [ mats] = getMatsfromNet(net )
% getMatsfrom net gets network and produces the matrix of linearities in a
% cell array, where each cell has a 3 dimensional matrix and each page of the
% matrix is the linear transformation of a block
mats = cell(numel(net.layers),1);
j = 1;
for i = 1 : numel(net.layers)
    
 Type = net.layers{i}.type;
 switch Type
     case 'conv'
         if isfield(net.layers{i},'blocks')
         blocks = net.layers{i}.blocks;
         else
             blocks = 1;
         end
         Weights = net.layers{i}.weights{1};
         Bias = net.layers{i}.weights{2};
         mats{j,1} = convMat2mat(Weights,blocks);
         mats{j,2} = Bias;
         j = j+1;
         
     otherwise
 end
end
end
function mat = convMat2mat(ConvMat,blocks)
    SZ = size(ConvMat);
    blockSize = SZ(4)/blocks;
    mat = reshape(ConvMat,[],SZ(4));
    mat = reshape(mat,size(mat,1),size(mat,2)/blocks,blocks);
    mat = permute(mat,[2,1,3]);
end

