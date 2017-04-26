function [ y_dzdx,dzdB,B] = em_nnmedian(x,w,l,dzdy,B)
%function [ y_dzdx,Bcell] = em_nnmedian(x,method,dzdy,B ,Bcell)
% deprecated: where Bcell is the of medians where the first cell is the values and the
% second is the logical locations
% current : B is the medians, with size 1,1,c,1
dzdB = [];
method = l.method;
if nargin> 3
    doder = true;
    if numel(dzdy(:,:,1,1))==1 && strcmp(method,'instance')
        y_dzdx = dzdy;
        B = [];
        dzdB = w.*0;
        return
    end
else
    doder = false;
    if numel(x(:,:,1,1))==1 && strcmp(method,'instance')
        % does nothing when the sample size is 1
        y_dzdx = x;
        B = [];
        dzdB = w.*0;
        return
    end
end

if ~doder
    
    %forward
    switch method
        case 'superMedian'
            vecsize = l.vectorSize;
            Set = l.set;
            
            xreshaped = reshape(x,[],1,1,size(x,4));
            chsz = size(xreshaped,3);
            y_dzdx = gpuArray.zeros(1,1,size(xreshaped,3)*l.vectorSize,size(x,4),'single');
            %xreshapedOdd = makeOdd(xreshaped);
            xmedrem = xreshaped;
           B = [];
            
            for i = 1:  vecsize
                meds = median(xmedrem,1,'omitnan');
                xmedrem = abs(xmedrem - meds);
                xmedrem(xmedrem ==0 )= nan;
                %y_dzdx(:,:,(i-1)*chsz +1:i*chsz,:) = meds;
                
            end
        case 'instance'
            xreshaped = reshape(x,[],1,size(x,3),size(x,4));
            xreshapedOdd = makeOdd(xreshaped);
            B = median(xreshapedOdd,1);
            y_dzdx  = x - B;
            dzdB = [];
        case 'batch'
            
            xreshaped = channelflat(x,'forward');
            xreshapedOdd = makeOdd(xreshaped);
            B = median(xreshapedOdd,1);
            y_dzdx = x-B;
        case 'estimate'
            B = [];
            y_dzdx = x-w;
    end
    
else
    %backward
    switch method
        case 'superMedian'
            y_dzdx = x.*0;
           
            dzdB =w.*0;
        case 'estimate'
            B = [];
            y_dzdx = dzdy;
            dzdB = sign(w-x);
            dzdB = sum(sum(sum(dzdB,1),2),4);
        case 'instance'
            dzdb = -dzdy;
            dzdb = sum(sum(dzdb,1),2);
            loc = findMedLocation(x,B,method);
            y_dzdx = dzdy + (dzdb.*loc);
            dzdB = w.*0;
        case 'batch'

            dzdb = -dzdy;
           dzdb = sum(sum(sum(dzdb,1),2),4);
            loc = findMedLocation(x,B,method);
            y_dzdx = dzdy + dzdb.*loc;
            dzdB = B;
    end

end
end
function [Inds] = findMedLocation(x,B,method)
% grabs x and B,the medians
% depending on method the sizes vary
% Method : batch
% B is 1,1,c,1
% Inds are logical same size as x
% Method : instance
% B is 1,1,c,b
% Inds are logical same size as x
switch method
    case 'batch'
        Inds = x==B;
        numIncidents = sum(sum(sum(Inds,1),2),4);
        coocurence = numel(find(numIncidents>1));
        noocurence = numel(find(numIncidents<1));
        if coocurence > 0
%            warning('coocurence happened')
        end
        if noocurence > 0
%            warning('No ocurence happened')
        end
        
    case 'instance'
        Inds = x==B;
        numIncidents = sum(sum(Inds,1),2);
        coocurence = numel(find(numIncidents>1));
        noocurence = numel(find(numIncidents<1));
        if coocurence > 0
            warning('coocurence happened')
        end
        if noocurence > 0
            warning('No ocurence happened')
        end
end
end
function y = makeOdd(x)
    % x is the reshaped version of original data with dimensions
    % (h*w*b)*1*c*1
    % makeOdd adds one row to data to make it odd only if the rows are even.
    if mod(size(x,1),2) ==0
        Infs = gpuArray.ones(1,1,size(x,3),size(x,4));
        Infs = Infs .* inf;
        y = cat(1,Infs,x);
    else
        y = x;
    end
end
function y = channelflat(x,method)
%flats the for each channel using all the data in the batch, 
% depending on forward and backward the input data size is different
% method = forward:
% x is h*w*c*b ,  y is (h*w*b)*1*c*1
% mehtod = backward
% x is (h*w*b)*1*c*1 , y is h*w*c*b
sz = size(x);
switch method
    case 'forward'
        y =  permute(x,[1,2,4,3]);
        y = reshape(y,[],1,sz(3),1);
    case 'backward'
        y = permute(x,[1,2,4,3]);
        y = reshape(y,size(x,1),size(x,2),size(x,4),size(x,3));
        y = permute(y,[1,2,4,3]);
end
end

