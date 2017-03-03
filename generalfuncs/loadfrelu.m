function [ w] = resetConvs(net,freezefrelus )
%LOADFRELU Summary of this function goes here
%   Detailed explanation goes here
freluNum = 1;
for i = 1 : numel(net.layers)
    l = net.layers(i);
    switch l.type
        case 'conv'
            if i ==1
                factor = 0.01;
            else
                factor =0.05;
            end
           net.layers(i).weights =  {{0.01*randn(5,5,3,k*32, 'single'), zeros(1, k*32, 'single')}};
    end
end

end

