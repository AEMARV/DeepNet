function sintheta_dzdtheta = sinl1(theta,dzdsintheta)
if nargin<2
    doder = false;
else
    doder = true;
end
if ~doder
    % forward
    sintheta_dzdtheta = cosl1(gather(theta-(pi/2)));
else
    % backward
    sintheta_dzdtheta = cosl1(gather(theta-(pi/2)),dzdsintheta);
end
end