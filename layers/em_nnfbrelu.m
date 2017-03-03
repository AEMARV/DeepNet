function [y_dzdx,dzdw] = em_nnfbrelu(x,w,dzdy,freeze)
%% w is two columns with the same number of channels as x.
% first column is the positive slope and the second is the negative slope
if nargin>2
    doder= true;
else
    doder = false;
end
if ~doder
    % forward
   yp = em_nnfrelu(x,w);
   yn = em_nnfrelu(-x,w);
   y_dzdx = cat(3,yp,yn); 
else
   
    % backward
    [dzdxp,dzdwp] = em_nnfrelu(x,w,dzdy(:,:,1:end/2,:),freeze);
    [dzdxn,dzdwn] = em_nnfrelu(-x,w,dzdy(:,:,(end/2)+1:end,:),freeze);
    y_dzdx = dzdxp- dzdxn;
    dzdw = dzdwp+dzdwn;
end

end

