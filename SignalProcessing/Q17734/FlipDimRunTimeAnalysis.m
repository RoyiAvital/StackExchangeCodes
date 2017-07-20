
function [ ] = FlipDimRunTimeAnalysis(  )


numIterations   = 100;
maxDim          = 1000;

runTime = 0;

for ii = 1:maxDim
    vX = randn([ii, 1]);
    hRunTime = tic();
    for jj = 1:numIterations
        vY = flip(vX);
    end
    runTime = runTime + toc(hRunTime);
end

disp(['Run Time - ', num2str(runTime)]);

runTime = 0;

for ii = 1:maxDim
    vX = randn([ii, 1]);
    hRunTime = tic();
    for jj = 1:numIterations
        vY = flipdim(vX, 1);
    end
    runTime = runTime + toc(hRunTime);
end

disp(['Run Time - ', num2str(runTime)]);

end