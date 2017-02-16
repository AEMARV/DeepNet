function [ y_dzdx,aux ] = em_nnbirelu(x,dzdy,aux )
%EM_NNBIRELU Summary of this function goes here
%   Detailed explanation goes here
isStoch = false;
chSize = size(x,3);

if nargin<2
    if isStoch
    probs = sampler(vl_nnsigmoid(x));
    aux = probs;
    y_dzdx= cat(3,x.*probs,x.*(~probs));
    else
        xcat = cat(3,x,-x);
        y_dzdx = vl_nnrelu(xcat);
        aux = [];
    end
else
    if isStoch
    activeAll = cat(3,aux,~aux);
    y_dzdx = dzdy .*activeAll;
    y_dzdx = dzdy(:,:,1:end/2,:)+ y_dzdx(:,:,(end/2)+1:end,:);
    else
    xcat = cat(3,x,-x);
    y_dzdx = vl_nnrelu(xcat,dzdy);
    y_dzdx = y_dzdx(:,:,1:end/2,:) - y_dzdx(:,:,(end/2)+1:end,:);
    end
    
end

end

