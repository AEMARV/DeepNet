function [y_dzdx,dzdw] = em_nnfrelu(x,w,method,dzdy,freeze)
%% w is two columns with the same number of channels as x.
% first column is the positive slope and the second is the negative slope
if isempty(method)
    method = 'cart';
end
wc = compileWfrelu(w,method);
wpos = reshape(wc(:,1),1,1,[],1);
wneg = reshape(wc(:,2),1,1,[],1);
if nargin>3
    doder= true;
else
    doder = false;
end
if ~doder
    % forward
    xpos = vl_nnrelu(x);
    xneg = -vl_nnrelu(-x);
    ypos = bsxfun(@times,xpos,wpos);
    yneg = bsxfun(@times,xneg,wneg);
    y_dzdx = ypos + yneg;
    
else
       
    % backward
    y_dzdx = bsxfun(@times,wpos,vl_nnrelu(x,dzdy))+ bsxfun(@times,wneg,vl_nnrelu(-x,dzdy));
    dzdwpos = vl_nnrelu(x).*dzdy;
    dzdwneg = -vl_nnrelu(-x).*dzdy;
    dzdwpos = sum(sum(sum(dzdwpos,4),2),1);
    dzdwneg = sum(sum(sum(dzdwneg,4),2),1);
    dzdw = w;
    dzdw(:,1) = reshape(dzdwpos,[],1);
    dzdw(:,2) = reshape(dzdwneg,[],1);
    dzdw = compileWfrelu(w,method,dzdw);
    if freeze
        dzdw = dzdw*0;
    else
     
    end
    
end
end

