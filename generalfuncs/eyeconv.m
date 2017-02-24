function [ out ] = eyeconv( h,w,c,k,dumb)
%EYECONV Summary of this function goes here
%   Detailed explanation goes here
out = orth(randn(h*w*c,h*w*c,'single'));
    for i = 1: floor(k/(h*w*c))
        out = cat(2,out,orth(randn(h*w*c,h*w*c,'single')));
    end

out = out(:,1:k);
if size(out,2)~= k
    error('restart')
end
out = reshape(out,[h,w,c,k]);

end

