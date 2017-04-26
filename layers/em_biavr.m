function [ y_dzdx ,B] = em_biavr( x,dzdy,B )
%EM_BIAVR Summary of this function goes here
%   Detailed explanation goes here
if nargin<2
    doder = false;
else
    doder=true;
end
if ~doder
    switch method
        case 'instance'
        case 'batch'
            xreshaped = permute(x,[1,2,4,3]);
            xreshaped = reshape(xreshaped,[],1,size(x,3),1);
            B = median(xreshaped,1);
            y_dzdx = bsxfun(@minus,x,B);
            y_dzdx = abs(y_dzdx)/2;
            y_dzdx = cat(3,y_dzdx,sign(x-B));
    end
else
    switch method
        case 'instance'
        case 'batch'
            dzdy = dzdy(:,:,1:end/2,:);
            y_dzdx = sign(x-B).*dzdy/2;
            dzdb = sign(B-x).*dzdy/2;
            dzdb = sum(sum(sum(dzdb,2),1),4);
            [Map] = findMedLocation(x,B,dzdb);
            Normalize = sum(sum(sum(Map,1),2),4);
            y_dzdx = (Map.*dzdb./Normalize) + y_dzdx;
    end
end
end
function [Inds] = findMedLocation(x,B,dzdb,method)
switch method
    case 'batch'
        dist = abs(x-B);
        distreshaped = permute(dist,[1,2,4,3]);
        distreshaped = reshape(distreshaped,[],1,size(x,3),1);
        [m,~] = min(distreshaped,[],1);
        Inds = distreshaped == m;
        Inds = permute(Inds,[1,2,4,3]);
        Inds = reshape(Inds,size(x,1),size(x,2),size(x,4),size(x,3));
        Inds = permute(Inds,[1,2,4,3]);
        
    case 'instance'
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
            gavg = vl_nnpool(x,[size(x,1),size(x,2)]);
            y_dzdx = bsxfun(@times,y_dzdx,gavg);
    end
else
    switch method
        case 'gavg'
            dtandx = 2*vl_nnsigmoid(x,dzdy);
            
            
    end
end
end
