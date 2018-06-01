function [ mB ] = ApplyGaussianBlur( mI, gaussianKernelStd, stdToRadiusFactor )

gaussianBlurRadius  = ceil(stdToRadiusFactor * gaussianKernelStd);

vGaussianKernel = exp(-((-gaussianBlurRadius:gaussianBlurRadius) .^ 2) / (2 * gaussianKernelStd * gaussianKernelStd));
vGaussianKernel = vGaussianKernel / sum(vGaussianKernel);

mI   = PadArrayReplicate(mI, gaussianBlurRadius);

mB = conv2(vGaussianKernel, vGaussianKernel.', mI, 'valid');


end

