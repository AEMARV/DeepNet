function [ Meds ] = GeoMed(x,w,isfirst )
%GEOMED Summary of this function goes here
%   Detailed explanation goes here
%% Constants
    Norm = 2;
    if nargin <3
        isfirst = true;
    end

%%---------------------------
    %w = constructBasis(w,Norm);
    curw = w(:,1);
    if norm(curw,1) ==0
        currM = sum(x.^2,2);
        Med = max(currM,[],1);
    else
    Measures = x*curw;
    currM = Measures(:,1);
    %currM = Oddize(currM);
    Med = median(currM,1,'omitnan');
    end
    [MinDev,~] = min(abs(currM-Med),[],1);
    MedsInd = find(abs(currM-Med)==MinDev);
    Meds = mean(x(MedsInd,:),1);
    return;
    goDeeper  = ~isUnique(x,MedsInd);
    if isnan(Med)
        Meds = x(MedsInd(1),:);
        return;
    end
    Mask = nan*zeros(size(currM),'like',currM);
    Mask(MedsInd) = 1;
    if goDeeper
        x = x .* Mask;
        warning('needs debug')
        if isfirst
            Meds = GeoMed(x,w(:,1:end),false);
        else
            Meds = GeoMed(x,w(:,2:end),false);
        end
    else
        Meds = x(MedsInd(1),:);
    end
    if isnan(Meds(1))
        warning('')
    end
end
function Measures = Oddize(Measures)
    nonNans = ~isnan(Measures);
    NumCount = sum(nonNans,1,'omitnan');
    if (mod(NumCount,2) == 0) && (NumCount ~=0)
        Measures = cat(1,Measures,-inf);
    end
end
function flag = isUnique(x,inds)
xmeds = x(inds,:);
devs = xmeds - xmeds(1,:);
if sum(abs(devs(:)))~= 0
    flag = false;
else
    flag = true;
end
end
