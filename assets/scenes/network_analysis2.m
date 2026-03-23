% filepath: /Users/yeonsu/GitHub/rod-dynamics-3d/assets/scenes/network_analysis.m
clear; clc; close all;

filename = '/Users/yeonsu/GitHub/rod-dynamics-3d/build/network_nofade.csv';

% filename = '/Users/yeonsu/GitHub/rod-dynamics-3d/build/network_test_zeroth.csv'; 

if ~isfile(filename), error('File not found: %s', filename); end

T = readtable(filename, 'VariableNamingRule','preserve');

frameCol = "frame";
rodICol  = "rod_i";
rodJCol  = "rod_j";

frames = unique(T.(frameCol));
targetFrame = frames(end);

% ---- moving-window settings ----
windowK = 100; % number of frames to aggregate (sliding window length)
% --------------------------------

targetIdx = find(frames == targetFrame, 1, 'last');
idx0 = max(1, targetIdx - windowK + 1);
winFrames = frames(idx0:targetIdx);

fprintf('Aggregating frames %d..%d (%d frames)\n', winFrames(1), winFrames(end), numel(winFrames));
F = T(ismember(T.(frameCol), winFrames), :);

% endpoints as strings (robust even if IDs are 0-based)
s = string(F.(rodICol));
t = string(F.(rodJCol));

% drop missing endpoints
mask = ~ismissing(s) & ~ismissing(t) & s ~= "NaN" & t ~= "NaN";
s = s(mask); t = t(mask);

% canonicalize undirected edges so (i,j) and (j,i) merge
swap = s > t;
s2 = s; t2 = t;
s2(swap) = t(swap);
t2(swap) = s(swap);

% count multiplicity per edge over the window (edge weight)
E = table(s2, t2, 'VariableNames', {'u','v'});
GC = groupcounts(E, ["u","v"]);           % GC.u, GC.v, GC.GroupCount
G = graph(GC.u, GC.v, GC.GroupCount);     % weighted undirected graph
G = simplify(G);                          % just in case

deg = degree(G);
w = G.Edges.Weight;

figure('Name', 'Moving-window Contact Network', 'Color','w');
p = plot(G);
p.NodeLabel = {};
p.NodeCData = deg;

if ~isempty(w) && max(w) > 0
    p.LineWidth = 0.2 + 3.0 * (w / max(w)); % thicker = more repeated contacts
else
    p.LineWidth = 0.5;
end

if ~isempty(deg) && max(deg) > 0
    p.MarkerSize = 2 + 6 * (deg / max(deg));
else
    p.MarkerSize = 3;
end

p.EdgeColor = [0.5 0.5 0.5];
p.EdgeAlpha = 0.4;
colormap(parula); colorbar;
title(sprintf('Contact Network (window=%d frames, %d..%d) | Nodes=%d Edges=%d', ...
    windowK, winFrames(1), winFrames(end), numnodes(G), numedges(G)));

figure('Name', 'Degree Distribution (Windowed)', 'Color','w');
histogram(deg);
xlabel('Number of Contacts (degree)'); ylabel('Count of rods');
grid on;