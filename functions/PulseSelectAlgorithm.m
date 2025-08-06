classdef PulseSelectAlgorithm
    % PulseSelectAlgorithm - defines enum for pulse selection algorithm
    % used in processSourceDetectorPair(...), loadWFmatTDmean(...) and
    % loadWFmatFDmean(...) functions.
    %
    % FixedDelayTrigger - selects the start of the laser and target signals
    %                     based on the fixed delay in nanoseconds from the
    %                     electric trigger send by laser driver to
    %                     oscilloscope
    % FixedDelayLaser   - specifies the fixed delay in nanosecond from the
    %                     laser pulse to the target pulse; laser pulse is
    %                     selected by cutoff std
    
    enumeration
        FixedDelayTrigger, FixedDelayLaser
    end
end

