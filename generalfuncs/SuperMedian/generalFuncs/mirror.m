function [xm,bin,Med] = mirror(x,v,Norm)
% mirrors x along v
% Bias shows the location of folding, if is empty the space is folded at
% median
% doCenter centers xm at Bias, the default value is false
% bin is the binary values N*1*1*b
% x is N*c*1*b and v is 1*c*1*b

%% Deprecated
%Medx =median(x,1,'omitnan');
%x = x - Medx;
% Mx = mean(x,1,'omitnan');
% Mx = Mx./sqrt(sum(Mx.^2,2));
% v = replacezv(v,Mx);
%v = v./sqrt(sum((v.^2 ),2));
Norm = 0;
%% End Deprecated
switch Norm
    case 0
        xnozero = x;
        xnozero(x==0) = nan;
        Med = median(xnozero,1,'omitnan');
        xc = x - Med;
        s = sign(xc);
        bin = setSign(s);
        xm = bin.*xc;
    case 1
        
xc = x- v;
Measure  = sum(xc.*sign(v),2);
bin = sign(Measure);
xm = bin .* xc;
    case 2
        Measure = x.*v;
        Measure = sum(Measure,2);
        Medval = median(Measure,1,'omitnan');
        Med = medianvec(x,Measure);
        bin = sign(Measure-Medval);
        binp = bin;
        binp(binp ==0 )  = 1;
        xc = x-Med;
        xm = binp.*xc;
end
% dotp = xcentered.*v;
% Measure = sum(dotp,2);
% % Measure is N*1*1*b
% Sign = sign(Measure);
% Sign(Sign == 0) = 1;
% MNan = Measure;
% %MNan(abs(MNan)<Thresh) = nan;
% Bias = median(MNan,1,'omitnan');
% MedDev = MNan - Bias;
% bin = sign(MedDev);
% Bias = v;
% % MedDev is N*1*1*b
% coef = abs(MedDev) - MedDev;
% coef(isnan(coef)) = 0;
% %CenterV = medianvec(MNan,x);
% xcentered = x - v;
% 
% xm = xcentered + coef.*v;
% xm(abs(xm)<Thresh) = 0;
 visualMirror(x,Med,xm,1,v)
 
end
function b = setSign(x)
%method= 'firstnz'
method = 'leastZeros'
switch method
    case 'firstnz'
        c = x~=0;
        [~,I] = max(c,[],2);
        [Rind,~,chInd,BInd] = ndgrid(1:size(x,1),1,1:size(x,3),1:size(x,4));
        LinInd = sub2ind(size(x),Rind,I,chInd,BInd);
        b = x(LinInd);
    case 'leastZeros'
        
end
end
function [] = visualMirror(x,CenterV,xm,sampleid,v)
avgNorm = 1;%max(sum(x(:,:,:,sampleid).*v(:,:,:,sampleid),2,'omitnan'),[],1,'omitnan');
vx = cat(1,v(1,1,1,sampleid),-v(1,1,1,sampleid));
vy = cat(1,v(1,2,1,sampleid),-v(1,2,1,sampleid));
vz = cat(1,v(1,3,1,sampleid),-v(1,3,1,sampleid));
figure(1);title('before')
scatter3(x(:,1,:,sampleid),x(:,2,:,sampleid),x(:,3,:,sampleid),'.')
hold on
scatter3(CenterV(:,1,:,sampleid),CenterV(:,2,:,sampleid),CenterV(:,3,:,sampleid),'red')
line(avgNorm.*vx,avgNorm.*vy,avgNorm.*vz,'Color','red')
hold off
drawnow;
figure(2);title('after')

scatter3(xm(:,1,:,sampleid),xm(:,2,:,sampleid),xm(:,3,:,sampleid),'.')
line(avgNorm*vx,avgNorm*vy,avgNorm*vz,'Color','red')
drawnow;
end