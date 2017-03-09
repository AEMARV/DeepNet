function cart_dzdtheta = l1phase2cart(theta,dzdcart)
% gets theta as n*1 input and output the cartesian coordinates in l2 which
% is n*2
if nargin < 2
    doder = false;
else
    doder = true;
end
if ~doder
    %forward
    costheta = cosl1(theta);
    sintheta = sinl1(theta);
    cart_dzdtheta = cat(2,costheta,sintheta);
else
    %backward
    dzdcos = dzdcart(:,1);
    dzdsin = dzdcart(:,2);
    cart_dzdtheta = cosl1(theta,dzdcos) + sinl1(theta,dzdsin);
    
end
end