% filepath: /Users/yeonsu/GitHub/rod-dynamics-3d/assets/scenes/network_analysis3.m
clear; clc; close all;

% --- Configuration ---
networkFile = '/Users/yeonsu/GitHub/rod-dynamics-3d/build/network_nofade.csv';
perrodFile  = '/Users/yeonsu/GitHub/rod-dynamics-3d/build/perrod_nofade.csv';
targetFrame = []; % Set to desired frame or use [] for last common frame
% ---------------------

if ~isfile(networkFile), error('File not found: %s', networkFile); end
if ~isfile(perrodFile),  error('File not found: %s', perrodFile); end

fprintf('Reading network data...\n');
Tn = readtable(networkFile, 'VariableNamingRule','preserve');

fprintf('Reading per-rod data...\n');
Tp = readtable(perrodFile, 'VariableNamingRule','preserve');

% Find common frames
framesN = unique(Tn.frame);
framesP = unique(Tp.frame);
commonFrames = intersect(framesN, framesP);

if isempty(commonFrames)
    error('No common frames found between network and per-rod files.');
end

% Auto-select last frame if not specified
if isempty(targetFrame)
    targetFrame = commonFrames(end);
else
    if ~ismember(targetFrame, commonFrames)
        fprintf('Warning: Frame %d not found in both files. Available common frames: %d..%d\n', ...
            targetFrame, min(commonFrames), max(commonFrames));
        targetFrame = commonFrames(end);
        fprintf('Switching to last common frame: %d\n', targetFrame);
    end
end

fprintf('Analyzing Force Balance at Frame %d\n', targetFrame);

% Filter data for the target frame
Fn = Tn(Tn.frame == targetFrame, :);
Fp = Tp(Tp.frame == targetFrame, :);

if isempty(Fn) || isempty(Fp)
    error('No data found for frame %d', targetFrame);
end

% Get list of all rods in this frame from per-rod data
rodIDs = Fp.rod;
numRods = length(rodIDs);

% Initialize force sums (Fx, Fy, Fz)
netContactForce = zeros(numRods, 3);
netFrictionForce = zeros(numRods, 3);

% Extract columns once (vectorized) instead of row-by-row access
rod_i = Fn.rod_i;
rod_j = Fn.rod_j;

Fax = Fn.force_a_x; Fay = Fn.force_a_y; Faz = Fn.force_a_z;
Fbx = Fn.force_b_x; Fby = Fn.force_b_y; Fbz = Fn.force_b_z;

Frax = Fn.friction_a_x; Fray = Fn.friction_a_y; Fraz = Fn.friction_a_z;

% Robustly find friction_b columns
Frbx = getCol(Fn, {'friction_b_x', 'friction_b.x'});
Frby = getCol(Fn, {'friction_b_y', 'friction_b.y'});
Frbz = getCol(Fn, {'friction_b_z', 'friction_b.z'});

% --- Sum forces (Vectorized Loop) ---
% Pre-map rod IDs to indices for speed
rodMap = containers.Map(rodIDs, 1:numRods);

for k = 1:height(Fn)
    idA = rod_i(k);
    idB = rod_j(k);
    
    if isKey(rodMap, idA)
        idxA = rodMap(idA);
        netContactForce(idxA, :)  = netContactForce(idxA, :)  + [Fax(k), Fay(k), Faz(k)];
        netFrictionForce(idxA, :) = netFrictionForce(idxA, :) + [Frax(k), Fray(k), Fraz(k)];
    end
    
    if isKey(rodMap, idB)
        idxB = rodMap(idB);
        netContactForce(idxB, :)  = netContactForce(idxB, :)  + [Fbx(k), Fby(k), Fbz(k)];
        netFrictionForce(idxB, :) = netFrictionForce(idxB, :) + [Frbx(k), Frby(k), Frbz(k)];
    end
end

totalNetForce = netContactForce + netFrictionForce;

% --- Compare with Gravity / Dynamics ---
gravity = [0, 0.0, 0]; % Adjust if your sim uses different gravity
mass = 1.0;              % Adjust if mass varies per rod

F_gravity = repmat(gravity * mass, numRods, 1);
F_residual = totalNetForce + F_gravity;

residualMag = sqrt(sum(F_residual.^2, 2));

% --- Visualization ---
figure('Name', 'Force Balance Analysis', 'Color', 'w');

subplot(2,2,1);
histogram(residualMag, 20);
title('Residual Force Magnitude Distribution');
xlabel('|F_{contact} + F_{gravity}|');
ylabel('Count');
grid on;

subplot(2,2,2);
scatter3(Fp.px, Fp.py, Fp.pz, 20, residualMag, 'filled');
colorbar;
title('Spatial Distribution of Force Imbalance');
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal; grid on;

subplot(2,2,3);
plot(totalNetForce(:,2), 'b.');
hold on;
yline(-gravity(2)*mass, 'r--', 'Weight');
title('Vertical Contact Force vs Weight');
xlabel('Rod Index'); ylabel('Force Y');
legend('Net Contact Fy', 'Rod Weight');
grid on;

subplot(2,2,4);
quiver3(Fp.px, Fp.py, Fp.pz, ...
        F_residual(:,1), F_residual(:,2), F_residual(:,3), ...
        0.5, 'r');
title('Residual Force Vectors');
axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');

fprintf('Mean Residual Force: %.4f\n', mean(residualMag));
fprintf('Max Residual Force:  %.4f\n', max(residualMag));

% --- Helper to safely get column data ---
function data = getCol(T, possibleNames)
    data = [];
    for i = 1:length(possibleNames)
        if ismember(possibleNames{i}, T.Properties.VariableNames)
            data = T.(possibleNames{i});
            return;
        end
    end
    error('Could not find column matching any of: %s', strjoin(possibleNames, ', '));
end