function optodeCoords = optodesRing(nOptodes,radius,cCoords,startAngle)
% optodesRing - generates 3D optode coordinates located on a ring in a
% cylindrical geometry
% 
% Arguments:
%   nOptodes    - number of optodes
%   radius      - radius of the ring
%   cCoords      - [x y z] coords of the optodes ring center
%   startAngle  - angle of the first optode in the ring
%
% Output:
%   optodeCoords - (nOptodes, 3) matrix of XYZ optode coords
%
% konstantin.tamarov@uef.fi

zCoords = cCoords(3) .* ones(nOptodes, 1);

thetaOptodes = startAngle + (0:-2*pi/nOptodes:-2*pi)';
thetaOptodes = thetaOptodes(1:nOptodes, 1);
optodeCoords = [...
    cCoords(1) + radius .* cos(thetaOptodes) ...
    cCoords(2) + radius .* sin(thetaOptodes) ...
    zCoords ...
    ];

end

