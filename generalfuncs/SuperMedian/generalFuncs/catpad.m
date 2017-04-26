function [ Bank] = catpad( dim,padval,Bank,batch )
%function [ Bank] = catpad( dim,padval,Bank,batch )
% 
if isempty(Bank)
    Bank = batch;
    return;
end
D1 = ndims(Bank);
D2 = ndims(batch);
D = max(D1,D2);
for i = 1 :D
    if i == dim
        continue;
    end
    [Bank,batch] = equalizeSize(Bank,batch,padval,i);

end
    Bank = cat(dim,Bank,batch);
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