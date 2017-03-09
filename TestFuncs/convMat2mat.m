function mat = convMat2mat(ConvMat,blocks)
% caluclates the matrix form of the convfilters. 
% ConvMat is h*w*c*k
% Blocks is the number of blocks in the Mat
% Output
% mat is a K/blocks * (h*w*C) * blocks tensor
    SZ = size(ConvMat);
    blockSize = SZ(4)/blocks;
    mat = reshape(ConvMat,[],SZ(4));
    mat = reshape(mat,size(mat,1),size(mat,2)/blocks,blocks);
    mat = permute(mat,[2,1,3]);
    vecSize = size(mat,2);
    Matfinal = zeros(size(mat,1),size(mat,2)*blocks);
    for i = 1 : blocks
        Ind = (i-1)*vecSize+1:i*vecSize;
        Matfinal(:,Ind)=  mat(:,:,i);
    end
    mat = Matfinal;
end
