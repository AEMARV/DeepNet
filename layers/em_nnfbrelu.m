function [y_dzdx,dzdw] = em_nnfbrelu(x,w,method,dzdy,freeze)
%% w is two columns with the same number of channels as x.
% first column is the positive slope and the second is the negative slope
if isempty(method)
method = 'polar2';
end
wc = compileWfBrelu(w,method);
wp = wc(:,1:2);
wn = wc(:,3:4);
if nargin>3
    doder= true;
else
    doder = false;
end
if ~doder
    % forward
   yp = em_nnfrelu(x,wp,'cart');
   yn  = em_nnfrelu(x,wn,'cart');
   y_dzdx = cat(3,yp,yn); 
else
   
    % backward
    dzdyp = dzdy(:,:,1:end/2,:);
    dzdyn = dzdy(:,:,(end/2)+1:end,:);
    [dzdxp,dzdwp] = em_nnfrelu(x,wp,'cart',dzdyp,freeze);
    [dzdxn,dzdwn] = em_nnfrelu(x,wn,'cart',dzdyn,freeze); 
    y_dzdx = dzdxp + dzdxn;
    dzdwc = cat(2,dzdwp,dzdwn); %Rotate
    dzdw = compileWfBrelu(w,method,dzdwc);% Rotate
    %dzdw = [dzdwp,dzdwn];

end

end


