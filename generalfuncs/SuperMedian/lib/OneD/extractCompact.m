function [MedianVals,MedianLocs,Bins,xmedrem] = extractCompact(x,Nanmap,pivot,calcbin,maxMed)
%% function [MedianVals,MedianLocs,Bins,xmedrem]  = extractMeds(x,NanMap,calcbin,maxMed)
% extractMeds calculate the Meds till every sample becomes NAN. 
% terms : N is element number, c is dimension size, b's are the set
% numbers, M is the median number, Bit is equal to M
% x is a d1,d2,d3,b,
% MedianVals is M,1,1,b,
% MedianLocs is a cell of size b,M where every cell has a matrix of size 
% M',3,1,b
% Bins are N,c,1,b,Bit
% NanMap determines how to code NANs in binary
% max med is the maximum number of rows or medians to extract
    MedianLocs = [];
    xmedrem = reshape(x,[],1,1,size(x,4));
    Bins = [];
    dynamic  = isempty(maxMed);
 
    if (calcbin && (ndims(x) ==5)) 
        error('Cannot calculate Binaries while given Binaries')
    end
    if ~dynamic
    MedianVals = zeros(maxMed,size(x,2),size(x,3),size(x,4),size(x,5));
    %MedianLocs = cell(size(x,4),maxMed);
    if calcbin
    Bins = zeros([size(x),maxMed])*Nanmap;
    end
    else
        MedianVals = [];
        MedianLocs = [];
    end
    i = 1;
    while(dynamic || i<=maxMed)
        meds = median(xmedrem,1,'omitnan');
        if dynamic
        MedianVals = cat(1,MedianVals,meds);
        else
            MedianVals(i,:,:,:) = meds;
        end
            curbin = sign(xmedrem-meds);
            curbin(curbin ==0) = Nanmap;
            curbin = reshape(curbin,size(x));
            locInds = duallocationSet(curbin,bitNum);
            LocCurrMed = extractMeds(locInds,nan,false,[]);
            MedianLocs = catpad(3,nan,MedianLocs,LocCurrMed);
            %LocMedCell = mat2cell(LocCurrMed,size(LocCurrMed,1),size(LocCurrMed,2),1,ones(1,size(x,4)));
            %LocMedCell = squeeze(LocMedCell);
%             if dynamic
%                 MedianLocs = cat(2,MedianLocs,LocMedCell);
%             else
%                 MedianLocs(:,i) = LocMedCell;
%             end
        if calcbin
            %curbin(isnan(curbin)) =Nanmap ;
            if dynamic
                Bins = cat(5,Bins,curbin);
            else
                Bins(:,:,:,:,i) = curbin;
            end
        end
        xmedrem = abs(xmedrem - meds);
        xmedrem(xmedrem <10^-1 )= nan;
        i = i+1;
        if ~hasnum(xmedrem)
            fprintf('\n %d Value medians extracted',i-1); 
            break;
        else
            %fprintf('\n iteration %d --- Compact',i)
        end
        
    end
end