function [ wn ] = maxMedpca( x,verbose )
%MAXMED Summary of this function goes here
%   Detailed explanation goes here
%% Constants
x = x - median(x,1,'omitnan');
Norm = 0;
method = 'disp';
covmat = true;
construct = true;
switch Norm
    case 0
        if construct
            xunit = createBasisL1(x)';
        else
            xunit = sign(x);
            xunit = sign(sum(xunit,2)).*xunit;
            xunit  = unique(xunit,'rows');
            Ind = find(isnan(xunit(:,1)),1);
            if ~isempty(Ind)
            xunit = xunit(1:Ind,:);
            end
        end
    case 1
        xunit = x./sum(abs(x),2);
    case 2
        xunit = x./sqrt(sum(x.^2,2));
end

if covmat
% Norms = sqrt(sum(x.*x,2));
% Norms = Norms * Norms';

% [~,MaxMedInd] = max(Meds,[],2);
% [m] = min(devs,[],1);
% MedsLogical = devs ==m;
% SelfMeds = MedsLogical .* eye(size(MedsLogical),'like',MedsLogical);
% SelfMeds = SelfMeds .*Norms;
% SelfMeds = sum(SelfMeds,1,'omitnan');
% [M,SelfMedind] = max(SelfMeds,[],2);
switch method
    case 'selfmed'
if M ==0
    wn = xunit(MaxMedInd,:)';
else
wn = xunit(SelfMedind,:)';
end
    case 'disp'
        cor = x*xunit';
        Meds = median(cor,1,'omitnan');
        devs = abs(cor - Meds) ;
        dispersions = sum(devs,1,'omitnan');
        [Max,maxdispInd] = max(dispersions,[],2);
        wn = xunit(maxdispInd,:)';
    case 'maxmed'
        wn = xunit(MaxMedInd,:)';
    case 'mostfreq'
        wn = findMaxOccurGrey(xunit)';
end
else
Norm = 2;
%% 
if nargin < 3
    verbose = false;
end
wn = rand(size(x,2),'like',x);
wn = wn./norm(wn,Norm)*0;
i = 0;
wac = wn;
while (true)
Med = GeoMed(x,wac);
wp = wn;
wn = Med';
if ismedian(x,wn)
    return;
end
% if inBank(wn,wac)
%     wac = zeros(size(wn),'like',wn);
%     wn = randn(size(wn),'like',wn);
% %     return;
% %     [~,wn] = inBank(wn,wac);
% %     wac = 0*wac;
% end
wac = cat(2,wn,wac);
if verbose
status(Med,wn,wp,i);
end
i = i+1;

end
end
end
function wac = createBasisL1(x)
Dim = size(x,2);
wac = eye(size(x,2),'like',x);
wp = (1:2^(Dim-1));
factor = (0:Dim-2)';
factor = 2.^factor;
wp = floor(wp./factor);
wp = mod(wp,2);
wp(wp ==0) = -1;
wp = cat(1,ones(1,size(wp,2),'like',x),wp);
wac = [wac,wp];
end
function flag = ismedian(x,wn)
flag = false;
Measure = x*wn;
wnMeasure = wn'*wn;
dev = Measure - wnMeasure;
signbias = sign(dev);
bias = sum(signbias,1,'omitnan');
if bias == 0
    flag = true;
    return 
end
end
function [flag,wavg] = inBank(w,wac)
if isnan(norm(w,1))
    wavg = w;
    flag = true;
    return;
end
flag = false;
diffs = wac - w;
diffs = sum(abs(diffs),1);
FirstInd = find(diffs==0,1);
if numel(FirstInd)>0
    flag = true;
    wavg = mean(wac(:,1:FirstInd),2);
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