function [ hLegendObject ] = ClickableLegend( varargin )


hLegendObject = legend(varargin{:});

set(hLegendObject, 'ItemHitFcn', @ToggleVisibility)


end


function [  ] = ToggleVisibility ( hLegendObj, sEventData )

switch(sEventData.Peer.Visible)
    case('on')
        sEventData.Peer.Visible = 'off';
    case('off')
        sEventData.Peer.Visible = 'on';
end


end

