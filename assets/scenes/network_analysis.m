% filepath : / Users / yeonsu / GitHub / rod - dynamics -
    % 3d / assets / scenes / network_analysis.m clear;
clc;
close all;

filename = '/Users/yeonsu/GitHub/rod-dynamics-3d/build/network_nofade.csv';
% if
%   ~isfile(filename) error('File not found: %s', filename);
% end

% Preserve original CSV headers T =
readtable(filename, 'VariableNamingRule', 'preserve');

frameCol = "frame";
rodICol = "rod_i";
rodJCol = "rod_j";

frames = unique(T.(frameCol));
targetFrame = frames(1);
% last frame fprintf('Analyzing frame: %d\n', targetFrame);

F = T(T.(frameCol) == targetFrame, :);

% Use node NAMES(strings)
so MATLAB
    doesn't require positive integer indices s = string(F.(rodICol)); t = string(
        F.(rodJCol));

% Drop missing / NaN endpoints mask = ~ismissing(s) & ~ismissing(t) &s ~ =
                                          "NaN" &t ~ = "NaN";
s = s(mask);
t = t(mask);

G = graph(s, t);
% undirected by default with edge list G = simplify(G);
% removes self - loops +
    parallel edges

        deg = degree(G);

figure('Name', sprintf('Contact Network - Frame %d', targetFrame), 'Color',
       'w');
p = plot(G);
p.NodeLabel = {};
% hide labels(usually too dense) p.NodeCData = deg;
if
  ~isempty(deg) && max(deg) > 0 p.MarkerSize = 2 + 6 * (deg / max(deg));
else
  p.MarkerSize = 3;
end p.EdgeColor = [0.5 0.5 0.5];
p.EdgeAlpha = 0.4;
colormap(parula);
colorbar;
title(sprintf('Rod Contact Network (Frame %d) | Nodes=%d Edges=%d',
              ... targetFrame, numnodes(G), numedges(G)));

figure('Name', 'Degree Distribution', 'Color', 'w');
histogram(deg);
xlabel('Number of Contacts (degree)');
ylabel('Count of rods');
title(sprintf('Degree Distribution (Frame %d)', targetFrame));
grid on;