close('all');

mX = readmatrix('mX.csv');
mY = readmatrix('mY.csv');

figure('Position', [100, 100, 1200, 800]);
% surf(mX, 'FaceColor', 'interp', 'EdgeColor', 'flat');
[vX, vY] = meshgrid(1:size(mX, 2), 1:size(mX, 1));
trisurf(delaunay(vX, vY), vX, vY, mX, 'FaceColor', 'interp', 'EdgeColor', 'flat');
% trisurf(delaunay(vX, vY), vX, vY, mX);
colormap(gca(), turbo);
xlabel('Wire');
ylabel('Time');
zlabel('X');
title('Estimated X');
view(45, 45);

ii = 2;

figure();
plot(mX(:, ii), 'DisplayName', 'X', 'LineWidth', 1.5);
hold('all');
plot(mY(:, ii), 'DisplayName', 'Y', 'LineWidth', 1.5);
xlabel('time');
ylabel('value');
title(['Signal: ', num2str(ii, '%03d')]);
ClickableLegend();
