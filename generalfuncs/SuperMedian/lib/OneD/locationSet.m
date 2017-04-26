function [locInds] = locationSet( Bin,pivot)
%% locInds = locationSet(Bin,pivot)
% creates a set of the coordinates where Bin is equal to pivot
% pivot is in [0,1,nan]
% Bin: 
% Bin is a d1,d2,d3,S, a 4 dimensional tensor
% locInds is d1*d2*d3,3,1,S first column is row, second is column and
% third is the channel index
% where d1,d2,d3 is the field, S is the sample 

sz = size(Bin);
Bin(Bin ~= pivot) = nan;
Bin(Bin == pivot) = 1;
[row,col,ch] = meshgrid(1:sz(2),1:sz(1),1:sz(3));
RowSet = Bin .* row;
RowSet = reshape(RowSet,[],1,1,sz(4));
ColSet = Bin.*col;
ColSet = reshape(ColSet,[],1,1,sz(4));
chSet = Bin.*ch;
chSet = reshape(chSet,[],1,1,sz(4));
locInds = cat(2,RowSet,ColSet,chSet);
locInds = pruneNan(locInds,sz(1),sz(2),sz(3));
end
function Inds = pruneNan(x,rowM,colM,chM)
xInd = sub2ind([rowM,colM,chM],x(:,1,:,:),x(:,2,:,:),x(:,3,:,:));
xIndSorted = sort(xInd,1);
firstNAN = find(isnan(sum(xIndSorted,4)),1);
if isempty(firstNAN)
    Inds = x;
    return;
end
Inds = xIndSorted(1:firstNAN-1,:,:,:);
[rowInds,colInds,chInds] = ind2sub([rowM,colM,chM],Inds);
Inds = cat(2,rowInds,colInds,chInds);
end

