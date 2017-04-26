function [ y_dzdx ] = em_biavr( x,dzdy )
%EM_BIAVR Summary of this function goes here
%   Detailed explanation goes here
if nargin<2
    doder = false;
else
    doder=true;
end
if ~doder
    y_dzdx = cat(3,x,x);
    y_dzdx(:,:,1:(end/2),:) = abs(x)/2;
   
    y_dzdx(:,:,(end/2)+1:end,:) =  vl_nnrelu(-x)/2;
else
    dzdxav = dzdy(:,:,1:(end/2),:)/2;
    dzdxtan = dzdy(:,:,(end/2) +1 :end,:);
    y_dzdx = sign(x).*dzdxav  -vl_nnrelu(-x,dzdxtan/2);
end
end
function y_dzdx = auxcalc(x,dzdy)
method = 'gavg'
if nargin > 1
    doder = true;
else
    doder = false;
end
if ~doder
    switch method
        case 'gavg'
            y_dzdx = 2*vl_nnsigmoid(x)-1;
            gavg = vl_nnpool(x,);
            y_dzdx = bsxfun(@times,y_dzdx,gavg);
    end
else
    switch method
        case 'gavg'
            
    end
end
end
