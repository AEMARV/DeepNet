function [wn ] = l1pca( x,thresh )
%L1PCA Summary of this function goes here
%   Detailed explanation goes here
%[w,d] = eig(cov(x));
if ~hasnum(x,false)
    wn = zeros(1,size(x,2),'like',x);
    return;
end
norm = 'l2';
wn = rand(size(x,2),'like',x);
%[wn,~] = eig(cov(x));
wn = wn(:,end);
wn = lNormalize(wn,norm);
wp = wn*nan;
wnorm = wn;
i = 1;
dw = zeros(size(wn))';
DevN = 0;
DevP = 10;
lambda = 1;
while (abs((DevN-DevP))>thresh || isnan(gather(sum((wn -  wp).^2))))
    measure = x*wn;
    %measure(abs(measure)<thresh) = nan;
    S = sign(measure - median(measure,1,'omitnan'));
    Minds = find(S ==0);
    if numel(Minds)~=0
    MindsRand = randperm(numel(Minds),numel(Minds)-1);
    S(Minds) = signat(x,Minds);
    end
    %S((abs(measure)<thresh)) = nan;
    DevP = DevN;
    DevN = mean(abs(measure- median(measure,1)),1,'omitnan');
   % fprintf('\n Deviation: %f',DevN);
    dwp = dw;
    dw = mean(S.*x,1,'omitnan');
    coef = 2^sign(dw*dwp');
    lambda = lambda*coef;
    lambda = min(lambda,2);
    %lambda = max(lambda,2^-10);
    wp = wn;
    wn = wn + dw'*lambda;
    wn = lNormalize(wn,norm);
    visualpca(x,wn,DevN);
end
wn = wn';
end
function [] = visualpca(x,v,DevN)
avgNorm = mean(sum(abs(x(:,:,:)),2,'omitnan'),1,'omitnan');
avgNorm = DevN;
vx = cat(1,v(1),-v(1));
vy = cat(1,v(2),-v(2));
vz = cat(1,v(3),-v(3));
gcf;
line(avgNorm*vx,avgNorm*vy,avgNorm*vz,'Color','green')
hold off
drawnow;
end
function y = signat(x,ind)
x = x(ind,:);
w = rand(size(x,2));

w = w(:,1);
w= lNormalize(w,'l1');
%w = lNormalize(w,'l2');
M = x*w;
S = sign(M - median(M,1));
%S(S==0) = sign(randn);
y = S;
end
function y = lNormalize(x,type)
switch type
    case 'l1'
        y = x./sum(abs(x),1);
    case 'l2'
        y = x./sqrt(sum(x.^2,1));
    case 'l3'
        y = x./(sum(x.^3,1)).^(1/3);
end
end