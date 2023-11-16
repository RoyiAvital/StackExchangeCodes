function [  ] = DisplayRunSummary( solverString, hObjFun, mX, runTime, cvxStatus )
% ----------------------------------------------------------------------------------------------- %
% Remarks:
%   1.  T
% Known Issues:
%   1.  A
% TODO:
%   1.  A
% Release Notes:
%   -   1.1.000     26/12/2020
%       *   Using MSE / Squared Norm.
%   -   1.0.000     23/11/2016
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    solverString {mustBeText}
    hObjFun (1, 1) {mustBeFunctionHandler}
    mX (:, :) {mustBeNumeric, mustBeReal}
    runTime (1, 1) {mustBeNumeric, mustBeReal, mustBePositive}
    cvxStatus {mustBeText} = {}
end

disp([' ']);
disp([solverString, ' Solution Summary']);
if(~isempty(cvxStatus))
    disp(['The ', solverString, ' Solver Status - ', cvxStatus]);
end
disp(['The Optimal Value Is Given By - ', num2str(hObjFun(mX))]);
disp(['The Optimal Argument Is Given By - [ ', num2str(mX(:).'), ' ]']);
drawnow(); %<! Make sure the buffer is emptied. A must when mX(:).' is very long
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


end


function [ ] = mustBeFunctionHandler( hF )
    % https://www.mathworks.com/matlabcentral/answers/107552
    if ~isa(hF, 'function_handle')
        eid = 'mustBeFunctionHandler:notFunctionHandler';
        msg = 'The 2nd input must be a Function Handler';
        throwAsCaller(MException(eid, msg));
    end
    
    
end

