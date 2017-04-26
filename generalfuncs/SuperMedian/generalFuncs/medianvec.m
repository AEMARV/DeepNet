function [ Median] = medianvec(x,Measure )
%MEDIANVEC Summary of this function goes here
%   Detailed explanation goes here
if nargin<2
    method = 'nomeasure';
else
method = 'random';
Med = median(Measure,1,'omitnan');
dev = abs(Measure-Med);
[m,Index] = min(dev,[],1);
Medlog = (dev==m) + 0;
Medlog(Medlog ==0) = nan;
centerv = Medlog .* x;
Median = centerv(1,:,:,:);

end

switch method
    case 'avg'
        Median = mean(centerv,1,'omitnan');
    case 'random'
        rowsInd = repmat(Index,1,size(x,2));
        index = 1:size(x,2);
        rscheme = size(rowsInd);
        rscheme(2) = 1;
        colInds = repmat(index,rscheme);
        index = 1:size(x,3);
        rscheme = size(rowsInd);
        rscheme(3) = 1;
        chInds = repmat(index,rscheme);
        index = reshape(1:size(x,4),1,1,1,[]);
        rscheme = size(rowsInd);
        rscheme(4) = 1;
        batchInds = repmat(index,rscheme);
        index = sub2ind(size(x),rowsInd,colInds,chInds,batchInds);
        Median = x(index);
    case 'nomeasure'
        Med = median(x,1,'omitnan');
        dev = abs(x - Med);
        dispersion = sum(dev,1,'omitnan');
        [m,Imeasure] = sort(dispersion,2,'descend');
        Median = medsorted(x,Imeasure);
        
end

end
function Med = medsorted(x,Imeasure)
    xi = x(:,Imeasure(1));
    M = median(xi,1,'omitnan');
    dev = abs(xi - M);
    MedsLoc = dev == min(dev,[],1);
    MedsInds = find(MedsLoc);
    NumMeds = numel(MedsInds);
    if NumMeds == 1
        Med = x(MedsInds,:);
    else
        x = x(MedsInds,:);
        Med = medsorted(x,Imeasure(2:end));
    end
end
