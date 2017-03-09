function costheta_dzdtheta = cosl1(theta,dzdcostheta)
if nargin< 2
    doder = false;
else
    doder = true;
end
if ~doder
    % forward
    costheta_dzdtheta = gpuArray(sawtooth(gather(theta+pi),0.5));
else
    % backward
    costheta_dzdtheta= gpuArray(dzdcostheta .* square(gather(theta+pi)));
end

end