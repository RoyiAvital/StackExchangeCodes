function [x,istop,lsmbError,sigFlag] = lsmb(A,b,lambda,tol,tau,conlim,itnlim,sigEst,show)

% LSMB  Iterative solver for least-squares problems
% 
% x = lsmb(A,b,lambda) returns the minimum-norm solution 
%     to the least-squares problem 
%     min_x ||A*x - b||_2^2 + lambda^2||x||^2
%
% Inputs
% ------
%   tol : Stopping tolerance. LSMB estimates the backward error
%         and stops when ERR(X)/NORM(A) < max(tol,eps)
%         Default value: 10^{-6}
%
%   tau : The backward error estimate depends on a parameter tau. 
%         The smaller tau is, the more backward error in b will 
%         be permitted relative to the error in A. 
%         If the user wants A and b to have relative backward errors
%         ALPHA and BETA, respectively, then a reasonable choice is
%         tol = ALPHA; tau = ALPHA*||A||/(BETA*||b||).
%         Default value: Inf
%
%   conlim: LSMB estimates the condition number of A and 
%           stops when CONDEST(A) > conlim. 
%           Default value: 10^{-8}
%
%   itnlim: Limit on the number of iterations. 
%           Default value: 2*min(m,n) if A is (m x n)
%   
%   sigEst: Any known lower bound on the smallest singular value of A.
%           Allows for more accurate error estimates. 
%           Note 1: If using some lambda =/= 0, make sigEst an estimate
%                   of sig_min(A), not sig_min([A ; lambda*I]). 
%           Note 2: To guard against instability, it is recommended to 
%                   underestimate sigEst by a small factor 
%                   (say, (1 - 10^{-10}))
%           Default value: 0
%
%   show : If 'true', periodically displays the error estimates. 
%          Default value: false
%
% Outputs
% -------
% x : The solution vector
% 
% istop : The reason for stopping. 
%         1: ERR(X)/NORM(A) < tol
%         2: CONDEST(A)     > conlim
%         3: ERR(X)/NORM(A) < eps
%         4: CONDEST(A)     > 1/eps
%         5: itn            > itnlim
%
% lsmbError : the absolute (not relative!) backward error
%
% sigFlag   : returns 0 if user input sigEst == 0 
%             returns -1 if the estimates using sigEst broke down
%             and returns 1 if they did not. 
%
% ----------------------------------------------------------------
% 02 Mar 2018: First release

% Much of this code was adapted from the Matlab implementation 
% of LSMR by D. C.-L. Fong and M. A. Saunders. 
% See http://web.stanford.edu/group/SOL/software/lsmr/

% Eric Hallman                 ehallman@berkeley.edu
% Department of Mathematics
% University of California, Berkeley
% ----------------------------------------------------------------

  u    = b;
  beta = norm(u);
  if beta > 0
    u  = u/beta;
  end
  
  v     = A'*u; 
  alpha = norm(v); 
  if alpha > 0
      v = v/alpha;
  end
  
  % Determine the dimensions (m,n) of A
  [m, n] = size(A); 
  minDim = min(m,n); 
  
  % Default parameters
  if nargin < 3 || isempty(lambda) , lambda   = 0;          end
  if nargin < 4 || isempty(tol)    , tol      = 1e-6;       end
  if nargin < 5 || isempty(tau)    , tau      = Inf;        end
  if nargin < 6 || isempty(conlim) , conlim   = 1e+8;       end
  if nargin < 7 || isempty(itnlim) , itnlim   = 2*minDim;   end
  if nargin < 8 || isempty(sigEst) , sigEst   = 0;          end
  if nargin < 9 || isempty(show)   , show     = false;      end
  
  showString = 'Iteration: %d, LSMB Error: %.3e\n';
  
  % Initialize variables for LSQR/LSMR
  itn    = 0; 
  rhoU   = alpha; 
  phiBar = beta; 
  rho    = 1; 
  rhoBar = 1;
  cBar   = 1; 
  sBar   = 0;  
  
  h      = v; 
  hBar   = 0;
  xC     = 0; 
  
  % Initialize variables for estimating ||r||
  rhoTildeU  = 1; 
  normRBase  = 0; 
  
  % Initialize variables for estimating ||x||
  zeta = 0; 
  zBar = alpha*beta; 
  zetaTilde = 0; 
  zetaHat = 0; 
  thetaTilde = 0; 
  normzHat = 0;
  cHat = 1; sHat = 0;
  
  % Initialize variable for carrying out the Cholesky factorization
  % in cases where we have an estimate of sigmaMin(A). 
  rhoShiftU = alpha; 
  rhoEst    = sigEst; 
  if sigEst > 0, sigFlag = 1; else, sigFlag = 0; end
 
  % Initialize variables for estimation of ||A|| and cond(A).
  normA2  = alpha^2;
  maxRhoBar = 0;
  minRhoBar = Inf;
  
  % Items for use in stopping rules
  istop  = 0; 
  ctol   = 0;         
  if conlim > 0, ctol = 1/conlim; end
  
  % Exit if b=0 or A'*b = 0.
  normAr = alpha * beta;
  if normAr == 0, return, end
  
  %------------------------------------------------------------------
  %     Main iteration loop.
  %------------------------------------------------------------------
  while itn < itnlim
    itn = itn + 1; 
    
    % Perform the next step of the bidiagonalization process. 
    % Satisfies the relations
    %      beta*u  =  A*v  - alpha*u,
    %      alpha*v  =  A'*u - beta*v. 
    
    u    = A*v - alpha*u; 
    beta = norm(u); 
    
    if beta > 0
        u     = u/beta; 
        
        v     = A'*u - beta*v; 
        alpha = norm(v); 
        
        if alpha > 0, v = v/alpha; end
    end
    % At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
    
    % ---------------------------------
    % Plane rotation Q_{i,2i+1} to handle regularization
    
    alphaReg = norm([rhoU lambda]);
    cReg     = rhoU/alphaReg;
    sReg     = lambda/alphaReg;
    
    normRBase = norm([normRBase sReg*phiBar]); 
    phiBar    = cReg * phiBar;   
    
    % ---------------------------------
    % Use a plane rotation (Q_i) to turn B_i to R_i.
    
    rhoOld = rho; 
    rho    = norm([alphaReg, beta]); 
    c      = alphaReg/rho; 
    s      = beta/rho;
    
    theta  = s * alpha; 
    rhoU   = c * alpha; 
    phi    = c * phiBar; 
    phiBar = -s * phiBar;
    
    % ---------------------------------
    % Use a plane rotation (QBar_i) to turn R_i^T to R_i^Bar.
    
    rhoBarOld = rhoBar;  
    thetaBar  = sBar * rho; 
    rhoBarU   = cBar * rho; 
    rhoTemp   = cBar*rho;    
    rhoBar    = norm([rhoBarU, theta]);
    cBar      = rhoBarU/rhoBar;  
    sBar      = theta/rhoBar;
    zetaOld   = zeta;
    zeta      = cBar*zBar; 
    zBar      = -sBar*zBar;
    
    thetaWideHat = sBar * rhoU; 
    rhoWideHat   = cBar * rhoU; 
    
    % ---------------------------------
    % Update hBar, h, x
    hBar = h - (thetaBar*rho/(rhoOld*rhoBarOld))*hBar;
    xC   = xC + (phi/rho)*h; 
    h    = v - (theta/rho)*h;
    
    
    % ---------------------------------
    % Use a plane rotation (Q_i^shift) to estimate rhoMin. 
    
    rhoShift = norm([rhoShiftU beta]); 
    cShift   = rhoShiftU/rhoShift; 
    sShift   = beta/rhoShift; 
    
    thetaShift = sShift * alpha; 
    rhoShiftU   = cShift * alpha; 
    
    sBreve = rhoEst/rhoShift; 
    if sBreve >= 1    % If sBreve >= 1 then sigEst cannot be used 
        sBreve = 0;   % as a reliable lower bound on smin(A). 
        sigEst = 0;   % This code abandons the estimate entirely. 
        sigFlag = -1; 
    end
    tBreve = sBreve/sqrt(1-sBreve^2); 
    rhoEst = norm([sigEst, thetaShift*tBreve]);
   
    rhoMin = max(rhoShiftU, rhoEst); 
    
    cMin        = rhoShiftU/rhoMin; 
    sMin        = sqrt(1-cMin^2); 
    betaMin   = sMin*rhoMin; 
    
    % rhoReg is our attempt to estimate the ||P_A(r)|| as
    % accurately as possible. If sigEst == 0, we have betaTilde = 0; 
    % If sigEst == 0 and lambda == 0, then rhoReg = rhoU, 
    % meaning that our estimate of ||P_A(r)|| will be equal to ||r||. 
    
    rhoCirc = norm([rhoU, betaMin, lambda]); 
    cErrEst   = rhoU/rhoCirc; 
    
    % ---------------------------------
    % Estimate ||r||
    
    thetaTildeOld = thetaTilde;
    rhoTilde      = norm([rhoTildeU thetaBar]);
    cTilde        = rhoTildeU/rhoTilde; 
    sTilde        = thetaBar/rhoTilde; 
    thetaTilde    = sTilde*rhoBar; 
    rhoTildeU     = cTilde*rhoBar; 
    residual      = phiBar*thetaWideHat/rhoTildeU;
    
    %||r|| = norm([normRBase, phiBar, gamma*residual])
    
    % ---------------------------------
    % Estimate ||x||
        
    rhoHatU  = cHat*rhoTilde;
    thetaHat = sHat*rhoTilde; 
    rhoHat   = norm([rhoHatU, thetaTilde]); 
    cHat     = rhoHatU/rhoHat; 
    sHat     = thetaTilde/rhoHat; 

    zetaTildeOld = zetaTilde; 
    zetaTilde    = (zetaOld - thetaTildeOld*zetaTildeOld)/rhoTilde;
    zetaTildeU   = (zeta - thetaTilde*zetaTilde)/rhoTildeU;

    rhoDDots   = norm([rhoHatU, sTilde*rhoBarU]); 
    cDDots     = rhoHatU/rhoDDots; 
    sDDots     = (sTilde*rhoBarU)/rhoDDots; 
    thetaDDots = sDDots*cTilde*rhoBarU; 
    rhoDDotsU  = cDDots*cTilde*rhoBarU;

    zetaHatOld = zetaHat; 
    zetaHat    = (zetaTilde - thetaHat*zetaHatOld)/rhoHat;
    zetaHatU   = (zetaTilde - thetaHat*zetaHatOld)/rhoDDots;
    zetaHatUU  = (zetaTildeU - thetaDDots*zetaHatU)/rhoDDotsU; 

    normzHat   = norm([normzHat zetaHatOld]); 
    normXBase  = norm([normzHat zetaHatU]);
    
    

    
    
    % --------------------------------
    % Solve a cubic equation to get a value of gamma
    % (therefore omegaTilde), and compute xLSMB
    rhoErrEst = rhoWideHat/cErrEst; 

    d1 = residual; 
    d2 = rhoErrEst/rhoDDotsU; 
    d3 = zetaHatUU*rhoErrEst - d1*d2;
    d4 = normXBase*rhoErrEst;
    normrC = norm([normRBase, phiBar]); 

    pol = [d1^2*(1+d2^2), d1*(2*d2*d3-d1), ...
                  d4^2+d3^2+normrC^2 + (rhoErrEst/tau)^2, -normrC^2];

    myPoly = @(x)polyval(pol,x); 
    gamma  = fzero(myPoly,0.5);
    gamma  = max(gamma,0); 
    gamma  = min(gamma,1); 
    
    % At this point we have
    % x  = xC + (gamma*phiBar*thetaWideHat/(rho*rhoBar))*hBar;
    % but we only need to compute it at convergence
    
    % ---------------------------------
    % Error estimates
    
    xresidual  = residual/rhoDDotsU;
    normxB = norm([normXBase, zetaHatUU + (gamma-1)*xresidual]);
    normrB = norm([gamma*residual, normrC]); 
    omegaB = omegaFun(normrB,normxB,tau);
    
    lsmrAtR   = abs(phiBar*rhoWideHat);     
    lsmbError = omegaB*lsmrAtR/norm([rhoCirc, omegaB])/normrB;
    
    % ---------------------------------
    % Estimate ||A||
    normA2        = normA2 + beta^2;
    normA         = sqrt(normA2);
    normA2        = normA2 + alpha^2;
    
    % ---------------------------------
    % Estimate cond(A).
    maxRhoBar       = max(maxRhoBar,rhoBarOld);
    if itn>1 
      minRhoBar     = min(minRhoBar,rhoBarOld);
    end
    condA         = max(maxRhoBar,rhoTemp)/min(minRhoBar,rhoTemp);
    
    % ---------------------------------
    % Stopping criteria
    
    test1 = lsmbError/normA; 
    test2 = 1/condA;

    % Stop if (lsmbError/normA < eps) or (condA > 1/eps).

    if itn >= itnlim,   istop = 5; end
    if 1 + test2  <= 1, istop = 4; end
    if 1 + test1  <= 1, istop = 3; end

    % Stop if (lsmbError/normA < tol) or (condA > conlim)
    if  test2 <= ctol,  istop = 2; end
    if  test1 <= tol,   istop = 1; end
    
    
    if istop > 0, break, end
    
    if show 
        if mod(itn,10) == 0
            fprintf(showString, itn, test1);
        end
    end
    
  end
  % End of main iteration loop.
  
  
  % At convergence, transfer from xC (LSQR) to x (LSMB).  
  x  = xC + (gamma*phiBar*thetaWideHat/(rho*rhoBar))*hBar;
  
  if show
      fprintf('Algorithm has converged.\n'); 
      fprintf(showString, itn, test1);
  end
  
end
%end of function


%------------------------------
% Nested functions
%------------------------------

function omega = omegaFun(rnorm,xnorm,tau)
    if tau > 1
        omega = rnorm/norm([1/tau, xnorm]);
    else
        omega = tau * rnorm / norm([1, tau*xnorm]);
    end
end

