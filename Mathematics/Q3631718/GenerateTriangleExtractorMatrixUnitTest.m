
clear();

EXTRACT_LOWER_TRIANGLE = 1;
EXTRACT_UPPER_TRIANGLE = 2;

INCLUDE_DIAGONAL = 1;
EXCLUDE_DIAGONAL = 2;

vNumRows = [2, 2 + sort(randperm(97, 25)), 100];
% vNumRows = [2, 3, 4];

for ii = 1:length(vNumRows)
    numRows = vNumRows(ii);
    
    triangleFlag = randi([1, 2], 1, 1);
    diagFlag = randi([1, 2], 1, 1);
    
    mX = randn(numRows, numRows);
    vX = mX(:);
    mLU = GenerateTriangleExtractorMatrix(numRows, triangleFlag, diagFlag);
    
    if((triangleFlag == EXTRACT_LOWER_TRIANGLE) && (diagFlag == INCLUDE_DIAGONAL))
        assert(isequal(mLU * vX, mX(logical(tril(mX, 0)))));
    end
    
    if((triangleFlag == EXTRACT_LOWER_TRIANGLE) && (diagFlag == EXCLUDE_DIAGONAL))
        assert(isequal(mLU * vX, mX(logical(tril(mX, -1)))));
    end
    
    if((triangleFlag == EXTRACT_UPPER_TRIANGLE) && (diagFlag == INCLUDE_DIAGONAL))
        assert(isequal(mLU * vX, mX(logical(triu(mX, 0)))));
    end
    
    if((triangleFlag == EXTRACT_UPPER_TRIANGLE) && (diagFlag == EXCLUDE_DIAGONAL))
        assert(isequal(mLU * vX, mX(logical(triu(mX, 1)))));
    end
end

%{
mA = reshape(1:(numRows * numRows), numRows, numRows);
mLU * mA(:)
%}
