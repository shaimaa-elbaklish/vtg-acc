clear all; close all; clc;

%% Load Results
load('results_for_matlab_plotting_minACC.mat');

Velocity_ACC = Velocity_ACC(2:end, :);
Velocity_VTG = Velocity_VTG(2:end, :);


%% Plotting Per Controller
start_idx = find(Time == 1050);
end_idx = find(Time == 1200);

fs1 = 12;
fs2 = 14;

figure;
tc = tiledlayout(1, 3);
title(tc, 'VTG ACC', 'FontSize', fs2);
Vel = Velocity_VTG;
SH = Spacing_VTG;
TH = SH./Vel;

nexttile(tc);
hold on;
plot(Time(start_idx:end_idx), Vel(:, start_idx:end_idx), 'LineWidth', 1.5);
plot(Time(start_idx:end_idx), Lead_Velocity(start_idx:end_idx), 'k--', 'LineWidth', 1.5);
hL = legend({'Follower 1', 'Follower 2', 'Follower 3', ' Follower 4', 'Leader'}, 'FontSize', fs1);
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel('Time (s)', 'FontSize', fs1);
ylabel('Velocity (m/s)', 'FontSize', fs1);
ylim([8 26]);

nexttile(tc);
hold on;
plot(Time(start_idx:end_idx), SH(:, start_idx:end_idx), 'LineWidth', 1.5);
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel('Time (s)', 'FontSize', fs1);
ylabel('Space Headway (m)', 'FontSize', fs1);
ylim([10 50]);

nexttile(tc);
hold on;
plot(Time(start_idx:end_idx), TH(:, start_idx:end_idx), 'LineWidth', 1.5);
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel('Time (s)', 'FontSize', fs1);
ylabel('Time Headway (s)', 'FontSize', fs1);
ylim([0.9 2.3]);

hL.Layout.Tile = 'south';
hL.Orientation = 'horizontal';

%% TTC and Energy Consumption
TTC_Data = (Spacing_Data - 5) ./ (Velocity_Data - Lead_Velocity);
TTC_Data(Velocity_Data <= Lead_Velocity) = NaN;
min_TTC_Data = min(TTC_Data, [], 2);

TTC_ACC = (Spacing_ACC - 5) ./ (Velocity_ACC - Lead_Velocity);
TTC_ACC(Velocity_ACC <= Lead_Velocity) = NaN;
min_TTC_ACC = min(TTC_ACC, [], 2);

TTC_VTG = (Spacing_VTG - 5) ./ (Velocity_VTG - Lead_Velocity);
TTC_VTG(Velocity_VTG <= Lead_Velocity) = NaN;
min_TTC_VTG = min(TTC_VTG, [], 2);


F0 = 213; F1 = 0.0861; F2 = 0.0027; mass = 1500;

VF_Data = Velocity_Data(:, 1:end-1);
Pt_Data = (F0 + F1.*VF_Data + F2.*VF_Data.^2 + 1.03.*mass.*Acceleration_Data).*VF_Data.*1e-03;
Pt_Data(Pt_Data < 0) = 0;
Ec_Data = sum(Pt_Data, 2) ./ (0.036 * sum(VF_Data, 2));

VF_ACC = Velocity_ACC(:, 1:end-1);
Pt_ACC = (F0 + F1.*VF_ACC + F2.*VF_ACC.^2 + 1.03.*mass.*Acceleration_ACC).*VF_ACC.*1e-03;
Pt_ACC(Pt_ACC < 0) = 0;
Ec_ACC = sum(Pt_ACC, 2) ./ (0.036 * sum(VF_ACC, 2));

VF_VTG = Velocity_VTG(:, 1:end-1);
Pt_VTG = (F0 + F1.*VF_VTG + F2.*VF_VTG.^2 + 1.03.*mass.*Acceleration_VTG).*VF_VTG.*1e-03;
Pt_VTG(Pt_VTG < 0) = 0;
Ec_VTG = sum(Pt_VTG, 2) ./ (0.036 * sum(VF_VTG, 2));

%% Equilibrium Speed
V_eq = zeros(size(Lead_Velocity));
step = 20/0.1;
for i = 1:step:length(Lead_Velocity)
    if i+step > length(Lead_Velocity)
        V_eq(i:end) = median(Lead_Velocity(i:end));
        break;
    else
        V_eq(i:i+step) = median(Lead_Velocity(i:i+step));
    end
    
end

%% Estimated L2 Gains
m = 100;
est_L2_gains_Data = estimate_L2_gain( ...
    V_eq', [Lead_Velocity; Velocity_Data]', m);
est_L2_gains_ACC = estimate_L2_gain( ...
    V_eq', [Lead_Velocity; Velocity_ACC]', m);
est_L2_gains_VTG = estimate_L2_gain( ...
    V_eq', [Lead_Velocity; Velocity_VTG]', m);

%% Figure TTC, Ec, GammaEst
n_platoon = 5;
fs1 = 12; fs2 = 14;
figure;
tcl = tiledlayout(1, 3);

nexttile(tcl);
hold on;
p = plot(1:n_platoon-1, est_L2_gains_Data, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, est_L2_gains_ACC, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, est_L2_gains_VTG, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel("Follower Order", 'FontSize', fs1);
ylabel('$\hat{\gamma}$', 'FontSize', fs1, 'Interpreter', 'latex');
hL = legend(["OpenACC Data", "CTG ACC", "VTG ACC"], 'Orientation','horizontal', 'AutoUpdate', 'off');

nexttile(tcl);
hold on;
p = plot(1:n_platoon-1, min_TTC_Data, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, min_TTC_ACC, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, min_TTC_VTG, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel("Follower Order", 'FontSize', fs1);
ylabel("Minimum Time To Collision (s)", 'FontSize', fs1);

nexttile(tcl);
hold on;
p = plot(1:n_platoon-1, Ec_Data, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, Ec_ACC, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
p = plot(1:n_platoon-1, Ec_VTG, 'diamond--', 'LineWidth', 1.0);
p.MarkerFaceColor = p.Color;
hold off;
grid on;
ax = gca;
ax.FontSize = fs1;
xlabel("Follower Order", 'FontSize', fs1);
ylabel("Tractive Energy Consumption (kWh/100km)", 'FontSize', fs1);

hL.Layout.Tile = 'south';
hL.FontSize = fs1;

%% Functions
function est_L2_gains = estimate_L2_gain(V_eq, V_platoon, m)
n_platoon = size(V_platoon, 2);
est_L2_gains = zeros(1, n_platoon-1);
for i = 2:n_platoon
    V_follow = V_platoon(:, i);
    V_lead = V_platoon(:, i-1);
    [~, Ru] = correlation_matrix(V_lead - V_eq, m);
    [~, Ry] = correlation_matrix(V_follow - V_eq, m);

    yalmip('clear');
    gainEst = sdpvar();
    obj = gainEst;
    cnst = [Ry - gainEst*Ru <= 0; gainEst >= 0];
    opts = sdpsettings('verbose', 1, 'solver', 'sedumi');
    optimize(cnst, obj, opts);

    est_L2_gains(i-1) = sqrt(value(gainEst));
end
end

function [Tm, Rm] = correlation_matrix(x, m)
N = length(x);
Tm = zeros(N+m-1, m);
for j = 1:m
    Tm(j:j+N-1, j) = x; 
end
Rm = (Tm' * Tm)/N;
end
