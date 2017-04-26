function [Medians,Bins,xmedrem] = extractMeds(x,Nanmap,calcbin,maxMed)
%% function [Meds,Bins] = extractMeds(x,NanMap)
% extractMeds calculate the Meds till every sample becomes NAN. 
% terms : N is element number, c is dimension size, b's are the set
% numbers, M is the median number, Bit is equal to M
% x is a N,c,1,b,
% Meds is M,c,1,b,
% Bins are N,c,1,b,Bit
% NanMap determines how to code NANs in binary
% max med is the maximum number of rows or medians to extract and is either
% a positive integer or empty indicating boundless.
    xmedrem = x;
    Bins = [];
    dynamic  = isempty(maxMed);
    if (calcbin && (ndims(x) ==5)) 
        error('Cannot calculate Binaries while given Binaries')
    end
    if ~dynamic
    Medians = zeros(maxMed,size(x,2),size(x,3),size(x,4),size(x,5));
    Bins = zeros([size(x),maxMed])*Nanmap;
    else
        Medians = [];
    end
    i = 1;
    while(dynamic || i<=maxMed)
        meds = median(xmedrem,1,'omitnan');
        if dynamic
        Medians = cat(1,Medians,meds);
        else
            Medians(i,:,:,:) = meds;
        end
        if calcbin
            curbin = sign(xmedrem-meds);
            curbin(curbin ==0) = Nanmap;
            %curbin(isnan(curbin)) =Nanmap ;
            if dynamic
                Bins = cat(5,Bins,curbin);
            else
                Bins(:,:,:,:,i) = curbin;
            end
        end
        xmedrem = abs(xmedrem - meds);
        xmedrem(xmedrem ==0 )= nan;
        i = i+1;
        if ~hasnum(xmedrem)
            %fprintf('\n %d medians extracted',i-1); 
            break;
        end
        
    end
end