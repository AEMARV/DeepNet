function [ locs] = duallocationSet( Bin,bitLoc )
%% locInds = locationSet(Bin,pivot)
% creates a set of the coordinates where Bin is equal to pivot
% pivot is in [0,1,nan]
% Bin: 
% Bin is a d1,d2,d3,S, a 4 dimensional tensor
% locInds is d1*d2*d3,1,4,S first column is row, second is column and
% third is the channel index
% where d1,d2,d3 is the field, S is the sample 
Bin(Bin ==0) = nan;
sz = size(Bin);
% Bin(Bin ~= pivot) = nan;
% Bin(Bin == pivot) = 1;
[row,col,ch] = ndgrid(1:sz(1),1:sz(2),1:sz(3));
RowSet = Bin .* row;
RowSet = reshape(RowSet,[],1,1,sz(4));
ColSet = Bin.*col;
ColSet = reshape(ColSet,[],1,1,sz(4));
chSet = Bin.*ch;
chSet = reshape(chSet,[],1,1,sz(4));
locInds = cat(2,RowSet,ColSet,chSet);
locs = pruneNan(locInds);
if isempty(locs)
    return
end
locs = cat(2,locs,ones(size(locs,1),1,1,size(locs,4))*bitLoc);
end
