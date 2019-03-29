function [ mD ] = CreateGradientOperator( numRows, numCols )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% mT = sparse(numRows, numCols);
mT = spalloc(numRows, numCols, 2 * (numRows - 1));

for jj = 1:numCols
    for ii = 1:numRows
        if(ii == numRows)
            break;
        end
        if(jj == ii)
            mT(ii, jj) = 1;
            mT(ii, (jj + 1)) = -1;
        end
    end
end

mD = [kron(eye(numRows), mT); kron(eye(numRows), mT.')];


end

