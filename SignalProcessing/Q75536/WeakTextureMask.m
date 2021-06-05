% Weak-Texture Mask generation
%
% msk = WeakTextureMask(img, patchsize, th)
%
%Output parameters
% msk: weak-texture mask. 0 and 1 represent non-weak-texture and weak-texture regions, respectively
%
%
%Input parameters
% img: input single image
% th: threshold which is output of NoiseLevel
% patchsize (optional): patch size (default: 7)
%
%Example:
% img = double(imread('img.png'));
% patchsize = 7;
% [nlevel th] = NoiseLevel(img, patchsize);
% msk = WeakTextureMask(img, patchsize, th);
% imwrite(uint8(msk*255), 'msk.png');
% version: 20150203


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noise Level Estimation:                                       %
%                                                               %
% Copyright (C) 2012-2015 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp                  %
%                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msk = WeakTextureMask(img, th, patchsize)

if( ~exist('patchsize', 'var') )
    patchsize = 7;
end

kh = [-1/2,0,1/2];
imgh = imfilter(img,kh,'replicate');
imgh = imgh(:,2:size(imgh,2)-1,:);
imgh = imgh .* imgh;

kv = kh';
imgv = imfilter(img,kv,'replicate');
imgv = imgv(2:size(imgv,1)-1,:,:);
imgv = imgv .* imgv;

numRows     = size(img, 1);
numCols     = size(img, 2);
numChannels = size(img, 3);

msk = zeros(numRows, numCols, numChannels);

for cha=1:numChannels
	m = im2col(img(:,:,cha),[patchsize patchsize]);
	m = zeros(size(m));
	Xh = im2col(imgh(:,:,cha),[patchsize patchsize-2]);
	Xv = im2col(imgv(:,:,cha),[patchsize-2 patchsize]);
    
	Xtr = sum(vertcat(Xh,Xv));
	
	p = (Xtr<th(cha));
	ind = 1;
	for col=1:numCols-patchsize+1
		for row=1:numRows-patchsize+1
			if( p(ind) > 0 )
				msk(row:row+patchsize-1, col:col+patchsize-1, cha) = 1;
			end
			ind = ind + 1;
		end
	end
	
end

end
