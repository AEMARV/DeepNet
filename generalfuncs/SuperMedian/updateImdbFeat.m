function [imdb] = updateImdbFeat(imdb,maxbit,maxloc )
%UPDATEIMDBFEAT Summary of this function goes here
%   Detailed explanation goes here
batchsize = 1000;
if isfield(imdb,'images')
    images = imdb.images.data;
else
    images = imdb.data;
end
Feat = [];
resBank = images;
imageNum = size(images,4);
MedianLocs = [];
for i =  1 : ceil(imageNum/batchsize)
    tic;
    fprintf('\nbatch no. %d/%d -------',i,ceil(imageNum/batchsize));
    [batch,Inds] = getBatch(resBank,i,batchsize);
    [newFeat,currLocs,~,~] = extractCompact( batch,nan,1,false,[]);
    currLocs = gather(currLocs);
    newFeat = gather(newFeat);
    Feat = gather(catpad(4,nan,Feat,newFeat));
    MedianLocs = gather(catpad(4,nan,MedianLocs,currLocs));
    T = toc;
    fprintf('ETA: %1.2d Hours -------',T*(ceil(imageNum/batchsize) - i)/3600);
end
if isfield(imdb,'images')
    imdb.images.valuefeature = Feat;
    imdb.images.locationfeature = MedianLocs;
else
    imdb.valuefeature = Feat;
    imdb.locationfeature= MedianLocs;
end


end
function [batch,Ind] = getBatch(images,i,batchsize)
Start = (i-1)*batchsize+1;
End = min(i*batchsize,size(images,4));
if Start > size(images,4)
    error('index is out of bound')
end
Ind = Start: End;
batch = gpuArray(images(:,:,:,Ind));

end
function Bank = catBatch(Bank,batch,padval)
[Bank,batch] = equalizeSize(Bank,batch,padval,1);
[Bank,batch] = equalizeSize(Bank,batch,padval,2);
[Bank,batch] = equalizeSize(Bank,batch,padval,3);
Bank = cat(4,Bank,batch);
end
function [v1,v2] = equalizeSize(v1,v2,padval,dim)
diff = size(v1,dim) - size(v2,dim);
if diff == 0
    return
end
padvec = size(v1)*0;
padvec(dim) = abs(diff);
if diff>0
    v2 = padarray(v2,padvec,padval,'post');
else
    v1 = padarray(v1,padvec,padval,'post');
end
end