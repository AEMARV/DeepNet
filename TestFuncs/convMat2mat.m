function mat = convMat2mat(ConvMat,blocks)
    SZ = size(ConvMat);
    blockSize = SZ(4)/blocks;
    mat = reshape(ConvMat,[],SZ(4));
    mat = reshape(mat,size(mat,1),size(mat,2)/blocks,blocks);
    mat = permute(mat,[2,1,3]);
end
