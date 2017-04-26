function [Meds ] = GeoMedL1( x,wac )
%GEOMEDL1 Summary of this function goes here
%   Detailed explanation goes here
wcurrent = wac(:,1)';

xc = x - wcurrent;
measure = xc.* sign(wcurrent);
measure = sum(measure,2);
measure = Oddize(measure);
Med = median(measure,1,'omitnan');
[~,MedsInd] = min(abs(measure-Med),[],1);
goDeeper  = ~isUnique(x,MedsInd);
Mask = nan*zeros(size(x(:,1)),'like',measure);
Mask(MedsInd) = 1;
if goDeeper && size(wac,2)>1
    xNull = x.*Mask;
    Meds = GeoMedL1(xNull,wac(:,2:end));
else
    Meds = x(MedsInd(1),:);
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
function Measures = Oddize(Measures)
    nonNans = ~isnan(Measures);
    NumCount = sum(nonNans,1,'omitnan');
    if (mod(NumCount,2) == 0) && (NumCount ~=0)
        Measures = cat(1,Measures,-inf);
    end
end