function [ wn] = pcal1( x,thresh)
%PCAL1 Summary of this function goes here
%   Detailed explanation goes here
cont = true;
wn = rand(size(x,2),1);
wn = Normalize(wn);
i = 0;
while(cont)
    i = i+1;
    wp = wn;
    measure = x*wn;
    S = sign(measure);
    wn = sum(S.*x,1,'omitnan')';
    wn = Normalize(wn);
    if ~iseqvec(wp,wn)
        continue;
    end
    if numel(find(abs(measure==0)))>0
        wn = wn + rand(size(wn))*0.00001;
        wn = Normalize(wn);
        continue;
    end
    break;
end
wn = wn';
end
function wn = Normalize(wn)
    wn = wn./sqrt(sum(wn.^2,1));
end
function flag = iseqvec(v1,v2)
diff = abs(v1-v2);
flag = sum(diff(:)) ==0;
end
