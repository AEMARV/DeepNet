function [ wn] = iterpcal1( x,verbose )
%MAXMED Summary of this function goes here
%   Detailed explanation goes here
%% Constants

%% 
if nargin < 2
    verbose = false;
end
j = 1;
%wn = sign(randn(size(x,2),1,'like',x));
notNan = find(~isnan(sum(x,2)));
if numel(notNan) ==0
    wn = x(1,:)';
    return;
end
wn = x(notNan(j),:)';
i = 0;
wac = wn;
while (true)
Med = GeoMedL1(x,wac);
wp = wn;
wn = Med';
 if norm(wp- wn,1)==0
     return;
 end
if inBank(wn,wac,inf)
    j = j+1;
    if j> numel(notNan)
        return;
    end
    wac = sign(x(notNan(j),:))';
else
    wac = cat(2,wn,wac);
end
if verbose
status(Med,wn,wp,i);
end
i = i+1;

end

end
function flag = inBank(w,wac,num)
if ~isinf(num)
wac = wac(:,1:num);
end
flag = false;
diffs = wac - w;
diffs = sum(abs(diffs),1);
if numel(find(diffs==0))>0
    flag = true;
end
end
function [] = status(Med,wn,wp,i)
fprintf('\n iteration %d - Median: ',i);
printvec(Med);
fprintf('|| PC: ');
printvec(wn);
fprintf('|| Median Projected Norm: %d', abs(wn'*Med'));

end
function [] = printvec(v)
fprintf('[')
for i = 1 : numel(v)
    fprintf('%f,',v(i));
end
fprintf(']')
end