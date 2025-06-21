clear();
close('all');
numSeasComp = 2;
vP = 10:20;
numSeasComp = length(vP);

tA = readtable('Signal.csv');
vX = tA.Time_s_;
vY = tA.RAWSignal;

% [vT, mS, vR] = trenddecomp(vY, 'NumSeasonal', numSeasComp);
[vT, mS, vR] = trenddecomp(vY, 'stl', vP);

cL = cell(1, 3 + numSeasComp);
cL{1} = 'Data';
cL{2} = 'Trend';
jj = 1;
for ii = 3:(3 + numSeasComp - 1)
    cL{ii} = ['Seasonality ', num2str(jj, '%02d')];
    jj = jj + 1;
end
cL{end} = 'Remainder';

plot([vY, vT, mS, vR]);
legend(cL);
