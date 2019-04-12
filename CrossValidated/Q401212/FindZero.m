function [ s ] = FindZero( f, a, b )

% This program tries to find a solution of f(x)=0 in the interval a<x<b.
% It's not required that f(a)f(b)<0, and it will report if a solution is not
% found in the interval.

% Usage (examples at end of file)
%   solve(@f,a,b)
%   solve(@(x)f,a,b)

% Required arguments:
%  f: the function f(x) is assumed to be defined using a function procedure,
%     or as an anonymous function (see examples at the end of this file)
%  a: lower endpoint
%  b: upper endpoint

% tol = error tolerance (can be adjusted as needed)
% tol=3*eps;
tol=1e-12;

% Imax = max number of secant steps allowed (can be adjusted as needed)
Imax=20;


% the code begins
if a >= b
    error('You need a < b')
end
if f(a)==0
    s=a;
    return
elseif f(b)==0
    s=b;
    return
end

% determine bounding interval for solution
[ind,x0,a,b]=sorty(f,a,b,19);
if isempty(ind)==1
    if f(x0)==0
        s=x0;
        return
    else
        [ind2,x0,a,b]=sorty(f,a,b,1999);
        if isempty(ind2)==1
            [ind3,x0,a,b]=sorty(f,a,b,42589);
            if isempty(ind3)==1
                error('Could not locate solution for a<x<b; try another interval')
            end
        end
    end
end
fL=f(a);
fR=f(b);

% avoid interval containing x=0
if a*b<0
    f0=f(0);
    if f0==0
        s=0;
        return
    elseif fL*f0<0
        b=0;
        fR=f0;
    else
        xL=0;
        fL=f0;
    end
end

% start curated secant method
iter=0;
err=10*tol+100;
xa=a; xb=b; fa=fL; fb=fR;
while err>tol
    xc=xb-fb*(xb-xa)/(fb-fa);
    % subdivide interval if secant starts to fail
    if xc<a || b<xc || abs(xc-xb)>abs(xb-xa)
        if xc<a || b<xc
            pts=[[a;xa;xb;b] [fL;fa;fb;fR]];
        else
            fc=f(xc);
            pts=[[a;xa;xb;xc;b] [fL;fa;fb;fc;fR]];
        end
        values2=sortrows(pts,1);
        for ii=1:length(pts(:,1))
            if pts(ii,2)*pts(ii+1,2)<0
                a=pts(ii,1);
                b=pts(ii+1,1);
                break
            end
        end
        [ind,x0,a,b]=sorty(f,a,b,19);
        fL=f(a); fR=f(b);
        xa=a; xb=b; fa=fL; fb=fR;
    else
        fc=f(xc);
        err=abs(xc-xb)/abs(xc);
        xa=xb;  fa=fb;
        xb=xc;  fb=fc;
        iter=iter+1;
        if fb==0
            s=xb;
            return
        end
        if iter>Imax
            error('Method failed: exceeded maximum number of iterations')
        end
    end
end
s=xb;

function [ind,x0,xa,xb]=sorty(f,a,b,nx)
xM=linspace(a,b,nx);
fp=zeros(nx,1);
for ix=1:nx
    fp(ix)=f(xM(ix));
end
values=[(1:nx-1)'  fp(1:nx-1).*fp(2:nx)];
values2=sortrows(values,2);
ind=find(values2(:,2)< 0,1,'last');
x0=xM(values2(2,1));
if isempty(ind)==1
    xa=a; xb=b;
else
    xa=xM(values2(ind,1));
    xb=xM(values2(ind,1)+1);
end


%%%% Example:  f=x^5-x^4-16  => x = 2

%%% Using anonymous function
% a=-5; b=5;
% v=solve(@(x)x^5-x^4-16,a,b)
% exact=2;
% error=(exact-v)/exact

%%% Using function file
% a=-5; b=5;
% v=solve(@f,a,b)
% exact=2;
% error=(exact-v)/exact
%
% function y=f(x)
% y=x^5-x^4-16;
% end

% Comments: the method requires f(x) to be continuous, and it
% can only find solutions where f(x) changes sign (so, it will fail to
% solve x^2=0).  Also, it might fail if there are an even number of
% solutions that are very close together; specifically, if they are closer
% than about 3e-5*(b-a) apart.

% Differences with MATLAB's fzero command:
%  a) solve does not require (or use) MATLAB's vectorization notation
%  b) solve is comparable with fzero in terms of speed and accuracy
%  c) solve can solve equations fzero can not solve (see examples.m file)

% Warning: The name 'solve' might conflict with a command in one of the
% toolboxes (which I don't have).  If so, just change its name.

%  Background
%  This was written for functions (or users) that do not lend themselves to
%  MATLAB's vector notation.  The algorithm is described in the text
%  "Introduction to Scientific Computing and Data Analysis" (Holmes, 2016).

%  Versions:
%  version 1.0: March 20, 2016
%  version 2.0: April 20, 2018 (simplified the code; improved the
%  refinement strategy; added example file)










