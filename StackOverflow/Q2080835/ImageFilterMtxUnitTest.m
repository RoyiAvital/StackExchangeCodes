
clear();

OPERATION_MODE_CONVOLUTION = 1;
OPERATION_MODE_CORRELATION = 2;

outputSizeString = 'same';

BOUNDARY_MODE_ZEROS         = 1;
BOUNDARY_MODE_SYMMETRIC     = 2;
BOUNDARY_MODE_REPLICATE     = 3;
BOUNDARY_MODE_CIRCULAR      = 4;

maxThr = 1e-9;

tic();
for numRowsImage = 22:26
    for numColsImage = 22:26
        
        mI = rand(numRowsImage, numColsImage);
        
        for numRowsKernel = 3:2:7
            for numColsKernel = 3:2:7
                
                mH = rand(numRowsKernel, numColsKernel);
                
                for operationMode = 1:2
                    
                    switch(operationMode)
                        case(OPERATION_MODE_CONVOLUTION)
                            convShapeString = 'conv';
                        case(OPERATION_MODE_CORRELATION)
                            convShapeString = 'corr';
                    end
                    
                    for boundaryMode = 1:4
                        
                        switch(boundaryMode)
                            case(BOUNDARY_MODE_ZEROS)
                                boundaryModeString = 0;
                            case(BOUNDARY_MODE_SYMMETRIC)
                                boundaryModeString = 'symmetric';
                            case(BOUNDARY_MODE_REPLICATE)
                                boundaryModeString = 'replicate';
                            case(BOUNDARY_MODE_CIRCULAR)
                                boundaryModeString = 'circular';
                        end
                        
                        mORef   = imfilter(mI, mH, boundaryModeString, outputSizeString, convShapeString);
                        mK      = CreateImageFilterMtx(mH, numRowsImage, numColsImage, operationMode, boundaryMode);
                        % mK      = CreateConvMtx(vH, numElementsSignal, convShape);
                        mO      = reshape(mK * mI(:), numRowsImage, numColsImage);
                        
                        disp([' ']);
                        disp(['Validating solution for the following parameters:']);
                        disp(['Image Size - [', num2str(numRowsImage), ' x ', num2str(numColsImage), ']']);
                        disp(['Kernel Size - [', num2str(numRowsKernel), ' x ', num2str(numColsKernel), ']']);
                        disp(['Boundary Mode - ', boundaryModeString]);
                        disp(['Convolution Shape - ', convShapeString]);
                        
                        mE = mO - mORef;
                        maxAbsDev = max(abs(mE(:)));
                        if(maxAbsDev >= maxThr)
                            disp([' ']);
                            disp(['Validation Failed']);
                            disp([' ']);
                        end
                        assert(maxAbsDev < maxThr);
                        
                    end
                    
                end
            end
        end
    end
end

toc();

