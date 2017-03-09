function cart_dzdtheta = l2phase2cart(theta,dzdcart)
% gets theta as n*1 input and output the cartesian coordinates in l2 which
% is n*2
    if nargin<2
        doder = false;
    else
        doder = true;
    end
    if ~doder
        %forward
        costheta = cos(theta);
        sintheta = sin(theta);
        cart_dzdtheta = cat(2,costheta,sintheta);
    else
        dzdcos = dzdcart(:,1);
        dzdsin = dzdcart(:,2);
        dcosdtheta = -sin(theta);
        dsindtheta = cos(theta);
        cart_dzdtheta = dzdcos.*dcosdtheta + dzdsin.*dsindtheta;
        %backward
    end
end