function [ w] = findMaxOccurGrey( xunit )
%FINDMAXOCCURGREY Summary of this function goes here
%   Detailed explanation goes here
%BitNum = size(xunit,2);
Sign = xunit(:,1);
Bits = xunit(:,2:end);
BitsFeat = Sign.*xunit;
[c,ia,ic] = unique(BitsFeat,'rows');
mostOccured = mode(ic);
w = c(mostOccured,:);

end

