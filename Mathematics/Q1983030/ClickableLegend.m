function [ hLegendObj ] = ClickableLegend( varargin )
% ----------------------------------------------------------------------------------------------- %
%[  ] = DetectAlgMain(  )
% Main function to run the 'Detect' algorithm
% Input:
%   - xxx               -   Input Paramseter.
%                           Structure: Matrix.
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
% Output:
%   - xxx               -   Constants Struct.
%                           Structure: Struct.
%                           Type: 'Single' / 'Double'.
%                           Range: [ ].

% References
%   1.  R
% Remarks:
%   1.  P
%   2.  C
% TODO:
%   1.  Adds
% Release Notes:
%   -   1.0.000    07/03/2016   Royi Avital
%       *   First realease version.
% ----------------------------------------------------------------------------------------------- %

% Create Legend
hLegendObj = legend(varargin{:});

set(hLegendObj, 'ItemHitFcn', @ToggleVisibility);


end


function [ ] = ToggleVisibility( hLegendObj, sEventData )

switch(sEventData.Peer.Visible)
    case('on')
        sEventData.Peer.Visible = 'off';
    case('off')
        sEventData.Peer.Visible = 'on';
end


end

