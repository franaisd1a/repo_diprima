function [ ra,dec ] = radecDeg2hgMS( ra_deg,dec_deg )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

dec=degrees2dms(dec_deg);
ra=degrees2dms(ra_deg/15);

% disp(sprintf('RaD  %10f H %10f M %10f S %10f',ra_deg,ra(1),ra(2),ra(3)));
% disp(sprintf('DecD %10f D %10f M %10f S %10f',dec_deg,dec(1),dec(2),dec(3)));

end

