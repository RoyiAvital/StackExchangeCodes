
% https://stackoverflow.com/questions/74697823
% https://stackoverflow.com/questions/14232991

close('all');

mB = imread('https://i.imgur.com/G3SdU3O.png');
mW = bwskel(mB);

mS = zeros(size(mB), 'logical');

mElement = strel([0, 1, 0; 1, 1, 1; 0, 1, 0]);
% mElement = strel([0, 1, 0; 1, 0, 1; 0, 1, 0]);

figure();
imshow(mB);

done = false();

while(~done)
    mE = imerode(mB, mElement);
    mT = imdilate(mE, mElement);
    mT = logical(mB - mT);
    mS = mS | mT;
    mB(:) = mE;

    numZeros = sum(mB, 'all');
    done = numZeros == 0;
end

figure();
imshow(mS);
figure();
imshow(mW);




