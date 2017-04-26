function [ValueMedians,Location] = extractMedPCA(orderedx,locDims,calcLoc,Norm,Thresh,pcaMethod)
%% function [ValueMedians,Location] = extractMedPCA(orderedx,locDims,calcLoc,Norm,Thresh)
% 
% extractMedPCA calculate folds the distribution at PCs till every sample becomes NAN . 
% terms : N is element number, c is dimension size, b's are the set
% numbers, M is the median number, Bit is equal to M
% x is a [locdims],c,b,
% ValueMedians is M,c,1,b,
% ---------------------------------------------------------------
% Location is the set of locations of 1 or 0 if calcloc is true
% Location is matrix of size L*locDims+1*b
    origSize = size(orderedx);
    x = createSet(orderedx,locDims);
    Bins = [];
    Location = [];
    ValueMedians = [];
    xmedrem = x;
    i = 1;
    while(true)
        PC = batchpca(xmedrem,1,pcaMethod,Thresh);
        [xmedrem,curbin,Meds] = mirror(xmedrem,PC,Norm);
        if calcLoc
        curbin = createOrderSet(curbin,locDims, origSize);
        curLocSet = duallocationSet(curbin,i);
        Location = cat(1,Location,curLocSet);
        Location = pruneNan(Location);
        end
        xmedrem = pluck(xmedrem,0,Norm);
        homo_prefix = cat(2,PC,Meds);
        ValueMedians = cat(1,ValueMedians,homo_prefix);
       
        i = i+1;
        if ~hasnum(xmedrem)
            fprintf('\n %d medians extracted',i-1); 
            break;
        else
            fprintf('\n %d medians extracted',i-1); 
        end
        
    end
    countBits = sum(Location,2);
    countBits = numel(find(~isnan(countBits)));
    countBits = countBits/size(x,4);
    avgBits = countBits/size(x,1);
    fprintf('\n %d Average BitRate', avgBits);
end
function Set = createSet(x,locationdim)
if locationdim <1
    Set = x;
    return;
end
if ndims(x) ~= locationdim +2
    error('There are spare dimensions')
end
Set = reshape(x,[],size(x,locationdim+1),1,size(x,locationdim+2));
end
function OrderSet = createOrderSet(Set,locationdim,sz)
    sz(locationdim+1) = 1;
    OrderSet = reshape(Set,sz);
end



function rem = pluck(rem,thresh,Norm)
temp = sum(abs(rem).^Norm,2).^(1/Norm);
temp(temp>thresh ) = 1;
temp(temp<=thresh) = nan;
rem = rem .* temp;
end