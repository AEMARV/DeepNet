function wc_dzdw = compileWfrelu(w,method,dzdwc)
if isempty(method)
    method = 'cart' ;
end
if nargin<3
    doder = false;
else
    doder = true;
end
if ~doder
    % forward
    switch method
        case 'cart'
            wc_dzdw = w;
        case 'cartconst'
            wc_dzdw = sign(w).*min(abs(w),1);
        case 'polar1'
            if size(w,2) ~= 1
                warning('unnecessary dimension in w');
            end
            wc_dzdw = l1phase2cart(w);
        case 'polar2'
            wc_dzdw = l2phase2cart(w);
    end
else
    %backward
    switch method
        case 'cart'
            wc_dzdw = dzdwc;
        case 'cartconst'
            wc_dzdw = dzdwc;
        case 'polar1'
            wc_dzdw = l1phase2cart(w,dzdwc);
        case 'polar2'
            wc_dzdw = l2phase2cart(w,dzdwc);
    end
end


end




