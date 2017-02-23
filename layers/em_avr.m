function [ y_dzdx ] = em_avr( x,dzdy )
%EM_AVR Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    y_dzdx = -abs(x);
else
    y_dzdx = -sign(x).*dzdy;
end

end

