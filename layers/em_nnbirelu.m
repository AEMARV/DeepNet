function [ y_dzdx,aux ] = em_nnbirelu(x,dzdy,aux )
%EM_NNBIRELU Summary of this function goes here
%   Detailed explanation goes here
isStoch = false;
chSize = size(x,3);
Scatter = true;
if Scatter
negInds = 2:2:2*chSize;
posInds = 1:2:2*chSize -1;
end

if nargin<2
    %regCheck(x);
    if isStoch
    probs = sampler(vl_nnsigmoid(x));
    aux = probs;
    y_dzdx= cat(3,x.*probs,x.*(~probs));
    else
        if Scatter
            xcat = cat(3,x,x);
            xcat(:,:,posInds,:) = x;
            xcat(:,:,negInds,:) = -x;
            y_dzdx = vl_nnrelu(xcat);
            aux = [];
        else
        xcat = cat(3,x,-x);
        y_dzdx = vl_nnrelu(xcat);
        aux = [];
        end
    end
else
    %regCheck(dzdy)
    if isStoch
    activeAll = cat(3,aux,~aux);
    y_dzdx = dzdy .*activeAll;
    y_dzdx = dzdy(:,:,1:end/2,:)+ y_dzdx(:,:,(end/2)+1:end,:);
    else
        if Scatter
            dydxpos = vl_nnrelu(x,dzdy(:,:,posInds,:));
            dydxneg = vl_nnrelu(x,dzdy(:,:,negInds,:));
            y_dzdx = dydxpos - dydxneg;
        else
            xcat = cat(3,x,-x);
            y_dzdx = vl_nnrelu(xcat,dzdy);
            y_dzdx = y_dzdx(:,:,1:end/2,:) - y_dzdx(:,:,(end/2)+1:end,:);
        end
    end
    
end

end

function y = regCheck(x)
x = x(:);
if ~isempty(find(isnan(x),1))
    error('has nan');
end
if ~isempty(find(isinf(x), 1))
    error('has inf');
end

end