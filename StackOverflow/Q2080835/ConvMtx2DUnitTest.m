
clear();

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

maxThr = 1e-9;

tic();
for numRowsImage = 28:32
    for numColsImage = 28:32
        
        mI = rand(numRowsImage, numColsImage);
        
        for numRowsKernel = 3:7
            for numColsKernel = 3:7
                
                mH = rand(numRowsKernel, numColsKernel);
                
                for convShape = 1:3
                    
                    switch(convShape)
                        case(CONVOLUTION_SHAPE_FULL)
                            numRowsOut = numRowsImage + numRowsKernel - 1;
                            numColsOut = numColsImage + numColsKernel - 1;
                            
                            convShapeString = 'full';
                        case(CONVOLUTION_SHAPE_SAME)
                            numRowsOut = numRowsImage;
                            numColsOut = numColsImage;
                            
                            convShapeString = 'same';
                        case(CONVOLUTION_SHAPE_VALID)
                            numRowsOut = numRowsImage - numRowsKernel + 1;
                            numColsOut = numColsImage - numColsKernel + 1;
                            
                            convShapeString = 'valid';
                    end
                    
                    mORef   = conv2(mI, mH, convShapeString);
                    % mK      = CreateConvMtx2D(mH, numRowsImage, numColsImage, convShape);
                    mK      = CreateConvMtx2DSparse(mH, numRowsImage, numColsImage, convShape);
                    mO      = reshape(mK * mI(:), numRowsOut, numColsOut);
                    
                    disp([' ']);
                    disp(['Validating solution for the following parameters:']);
                    disp(['Image Size - [', num2str(numRowsImage), ' x ', num2str(numColsImage), ']']);
                    disp(['Kernel Size - [', num2str(numRowsKernel), ' x ', num2str(numColsKernel), ']']);
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

toc();

