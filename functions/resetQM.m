function Geometry = resetQM(Geometry)
% resetQM - resets sources and detectors on the mesh
%
% Arguments:
%   Geometry    - struct, see createGeometry function
% 
% Output:
%   Geometry    - input with updated source and detector matrices on mesh
%
% konstantin.tamarov@uef.fi

% Create sources and detectors on the mesh
% Save the source and detector configuration to file
Geometry.hMesh.WriteQM([Geometry.dataDir 'QMfile.txt'], Geometry.sourceCoords, ...
    Geometry.detectorCoords, sparse(Geometry.measConfig));
% Setup the source and detectors on the mesh
Geometry.hMesh.ReadQM([Geometry.dataDir 'QMfile.txt']);
% See Toast help for parameter explanation
Geometry.qVec = Geometry.hMesh.Qvec(Geometry.sourceType, Geometry.sourceProfile, Geometry.sourceWidth);
Geometry.mVec = Geometry.hMesh.Mvec(Geometry.detectorProfile, Geometry.detectorWidth, Geometry.detectorRefInd);

end

