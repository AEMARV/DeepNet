function Inds = pruneNan(x)
% prunes rows of nan where the elements of x are integers and x N by d
% by... matrix
% Inds are the pruned rows.
sz = size(x);
minDim = min(x,[],1);
x = x - minDim +1;
maxDim = max(x(:));
if isnan(maxDim)
    Inds = [];
    return;
end
switch sz(2)
    case 1
        xInd = x;
    case 2
        xInd = sub2ind([maxDim,maxDim],x(:,1,:,:),x(:,2,:,:));
    case 3
        xInd = sub2ind([maxDim,maxDim,maxDim],x(:,1,:,:),x(:,2,:,:),x(:,3,:,:));
    case 4
        xInd = sub2ind([maxDim,maxDim,maxDim,maxDim],x(:,1,:,:),x(:,2,:,:),x(:,3,:,:),x(:,4,:,:));
    case 5
        xInd = sub2ind([maxDim,maxDim,maxDim,maxDim,maxDim],x(:,1,:,:),x(:,2,:,:),x(:,3,:,:),x(:,4,:,:),x(:,5,:,:));
end
xIndSorted = sort(xInd,1);
firstNAN = find(~(sum(xIndSorted,4,'omitnan')),1);
if isempty(firstNAN)
    Inds = x+minDim - 1;
    return;
end
Inds = xIndSorted(1:firstNAN-1,:,:,:);
Inds = gather(Inds);
switch sz(2)
    case 1
        Inds = xInd;
    case 2
        
        [d1Inds,d2Inds] = ind2sub([maxDim,maxDim],Inds);
        Inds = cat(2,d1Inds,d2Inds);
    case 3
        [d1Inds,d2Inds,d3Inds] = ind2sub([maxDim,maxDim,maxDim],Inds);
        Inds = cat(2,d1Inds,d2Inds,d3Inds);
    case 4
        [d1Inds,d2Inds,d3Inds,d4Inds] = ind2sub([maxDim,maxDim,maxDim,maxDim],Inds);
        Inds = cat(2,d1Inds,d2Inds,d3Inds,d4Inds);
    case 5
        [d1Inds,d2Inds,d3Inds,d4Inds,d5Inds] = ind2sub([maxDim,maxDim,maxDim,maxDim,maxDim],Inds);
        Inds = cat(2,d1Inds,d2Inds,d3Inds,d4Inds,d5Inds);
end
if numel(find(Inds(:)>maxDim)) >0
    error('gpu messup');
end
Inds = gpuArray(Inds + minDim - 1);
end
