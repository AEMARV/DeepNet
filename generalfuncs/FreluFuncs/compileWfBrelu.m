function wc_dzdw = compileWfBrelu(w,method,dzdwc)
phasediff = pi/2;
if nargin>2
    doder = true;
else
    doder = false;
end
if ~doder
    % forward
    switch method
        case 'polar2'
            if size(w,2)>1
                warning('theta has more dimensions');
            end
            theta = w(:,1);
            wp = l2phase2cart(theta);
            wn = l2phase2cart(theta + phasediff);
            wc_dzdw = cat(2,wp,wn)/2;
        case 'polar1'
            if size(w,2)>1
                warning('theta has more dimensions');
            end
            theta = w(:,1);
            wp = l1phase2cart(theta);
            wn = l1phase2cart(theta + phasediff);
            wc_dzdw = cat(2,wp,wn);
        case 'cartrot'
            %wp = sign(w).*min(abs(w),1);
            wp = w;
            wc_dzdw = [wp,rotate(wp)];
        case 'cartconst'
            wc_dzdw = sign(w).*min(abs(w),1);
        case 'cartl1rot'
            wp = polar2Cart(w,1);
            wrotated = w;
            wrotated(:,2) = w(:,2)+ (pi/2);
            wn = polar2Cart(wrotated,1);
            wc_dzdw = cat(2,wp,wn);
            
    end
else
    % backward
        dzdwp = dzdwc(:,1:2);
        dzdwn = dzdwc(:,3:4);
    switch method
        
        case 'polar2'
            theta = w(:,1);
            dzdw_P_dtheta = l2phase2cart(theta,dzdwp);
            dzdw_N_dtheta = l2phase2cart(theta+phasediff,dzdwn);
            wc_dzdw = dzdw_N_dtheta + dzdw_P_dtheta;
        case 'polar1'
            theta = w(:,1);
            dzdw_P_dtheta = l1phase2cart(theta,dzdwp);
            dzdw_N_dtheta = l1phase2cart(theta+phasediff,dzdwn);
            wc_dzdw = dzdw_N_dtheta + dzdw_P_dtheta;
        case 'cartrot'
            %w = sign(w).*min(abs(w),1);
            dzdwp = dzdwc(:,1:2);
            dzdwn = dzdwc(:,3:4);
            wc_dzdw = dzdwp + rotate(w,dzdwn);
        case 'cartconst'
            wc_dzdw = dzdwc;
        case 'cartl1rot'
            wrotated = w;
            wrotated(:,2) = w(:,2)+ (pi/2);
            dzdwrotated = polar2Cart(wrotated,1,dzdwn);
            dzdwnrotated = polar2Cart(w,1,dzdwp);
            wc_dzdw = dzdwrotated + dzdwnrotated;
            
    end
    
end
end
function cart_dzdw = polar2Cart(w,norm,dzdcart)
    if nargin>2
        doder = true;
    else
        doder = false;
    end
    if ~doder
        %forward
        r = w(:,1);
        theta = w(:,2);
        switch norm
            case 1
                x = r.* cosl1(theta);
                y = r.* sinl1(theta);
                cart_dzdw = cat(2,x,y);
            case 2
                x = r.*cos(theta);
                y = r.*sin(theta);
                cart_dzdw = cat(2,x,y);
        end
    else
        %backward
        dzdx = dzdcart(:,1);
        dzdy = dzdcart(:,2);
        r = w(:,1);
        theta = w(:,2);
        switch norm
            case 1
                dzdthetax = r.* cosl1(theta,dzdx);
                dzdthetay = r.* sinl1(theta,dzdy);
                dzdxr = dzdx.*cosl1(theta);
                dzdyr = dzdy.*sinl1(theta);
                dzdtheta = dzdthetax + dzdthetay;
                dzdr = dzdxr + dzdyr;
                cart_dzdw = cat(2,dzdr,dzdtheta);
            case 2
               error('not IMplemented');
        end
    end
end
function wr_dzdw=  rotate(w,dzdwrotate)
    if nargin >1
        doder = true;
    else
        doder = false;
    end
    
    if ~doder
        % forward pass
        wr_dzdw = w;
        wr_dzdw(:,1) = -w(:,2);
        wr_dzdw(:,2) = w(:,1);
    else
        % backward pass
        wr_dzdw = dzdwrotate;
        wr_dzdw(:,2) = -dzdwrotate(:,1);
        wr_dzdw(:,1) = dzdwrotate(:,2);
    end
end
function wr_dzdw = rotatel1(w,dzdwrotate)
if nargin <2
    doder = false;
else
    doder = true;
end
if ~doder
    %forward
    r= w(:,1);
    theta = w(:,2);
    
    
else
    %backward
end
end