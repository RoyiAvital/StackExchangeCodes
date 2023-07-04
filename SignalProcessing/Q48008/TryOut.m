
close('all');

mI = imread('RzlYW.png');
mB = imbinarize(mI);
mB = mB(:, :, 2);
mW = bwskel(mB);
figure(); imshow(mW);

mC = zeros(size(mW), 'logical');
mK = ones(3, 3);
mK(2, 2) = 0;

mN = conv2(mW, mK, 'same');
mSN = mN .* mW;
figure(); imagesc(mSN);

ll  = 0;
vII = zeros(numel(mC), 1);
vJJ = zeros(numel(mC), 1);

idxFirst = find(mSN == 1, 1, 'first');
[ii, jj] = ind2sub(size(mSN), idxFirst);
ll  = ll + 1;
vII(ll) = ii;
vJJ(ll) = jj;
mC(ii, jj) = 1;

[mm, nn] = FindNext(mW, mC, ii, jj);
ll  = ll + 1;
vII(ll) = mm;
vJJ(ll) = nn;
mC(mm, nn) = 1;

while(mSN(mm, nn) ~= 1)
    ii = mm;
    jj = nn;
    [mm, nn] = FindNext(mW, mC, ii, jj);
    ll  = ll + 1;
    vII(ll) = mm;
    vJJ(ll) = nn;
    mC(mm, nn) = 1;
end

middleIdx = round(ll / 2);
ii = vII(middleIdx);
jj = vJJ(middleIdx);

mI = repmat(double(mB), 1, 1, 3);
mI(ii, jj, :) = [1; 0; 0];

figure();
imshow(mI);



function [ mm, nn ] = FindNext(mW, mC, ii, jj)

for pp = -1:1
    mm = ii + pp;
    for qq = -1:1
        nn = jj + qq;
        % if((pp == 0) && (qq == 0))
        %     % Don't say put
        %     continue;
        % end
        if(mW(mm, nn) && ~mC(mm, nn))
            return;
        end
    end
end


end






