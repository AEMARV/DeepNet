function Ind = blockScatterInd(block,parity,Size)
%function Ind = blockScatterInd(block,parity,Size)
Inds = 1:Size;
IndsMat = reshape(Inds,block,[]);
offset = 1:(Size/block);
if parity == 0
offset = offset-1;
end
offset = offset * block;
Ind = bsxfun(@plus,IndsMat,offset);
Ind = Ind(:);


end