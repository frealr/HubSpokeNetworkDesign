close all;
clear all;
clc;

% === 1) RUTA de tu GAMS ===
gamsHome = 'C:\GAMS\50';  % <-- CAMBIA esto  % <-- CAMBIA esto

[basedir,~,~] = fileparts(mfilename('fullpath'));
basedir = fullfile(basedir, 'export_csv');   % subcarpeta de exportación
if ~exist(basedir,'dir'); mkdir(basedir); end

% === Tus datos ===

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_8node_network();

demand = demand./365;

M = 1e4;
nreg = 10;
eps = 1e-3;
vals_regs = linspace(0.005,0.995,nreg-1);
n_airlines = 5;
[lin_coef,bord,b] = get_linearization(n,nreg,alt_utility,vals_regs,n_airlines);

candidates = zeros(n);
for i=1:n
    candidates(i,candidasourcertes(i,:) > 0) = 1;
end

%%

% === Helpers para escribir CSV ===
write_matrix_csv = @(A, fn) writetable( ...
    array2table(A, ...
    'VariableNames', cellstr(compose('j%d', 1:size(A,2))), ...
    'RowNames',     cellstr(compose('i%d', (1:size(A,1))')) ), ...
    fullfile(basedir, fn), 'WriteRowNames', true );


write_vector_csv = @(v, fn, prefix) ( ...
    writetable( table( cellstr(prefix+string((1:numel(v)).')), v(:), ...
    'VariableNames', {'idx','value'} ), ...
    fullfile(basedir, fn)) ...
    );

write_scalar_csv_append = @(name, val, fn) ( ...
    writetable( table( string(name), double(val), ...
    'VariableNames', {'name','value'} ), ...
    fullfile(basedir, fn), 'WriteMode','append') ...
    );
% Initialize alfa_od and beta_od, gamma
alfa_od = ones(n);
beta_od = ones(n);
gamma = 20;

write_gams_param_iii('./export_txt/lin_coef.txt', lin_coef);
write_gams_param_iii('./export_txt/b.txt', b);
write_gams_param_iii('./export_txt/bord.txt', bord);

% === 2D matrices ===
write_gams_param_ii('./export_txt/demand.txt', demand);
write_gams_param_ii('./export_txt/travel_time.txt', travel_time);
write_gams_param_ii('./export_txt/alt_utility.txt', alt_utility);


write_gams_param_ii('./export_txt/link_cost.txt', link_cost);
write_gams_param_ii('./export_txt/link_capacity_slope.txt', link_capacity_slope);
write_gams_param_ii('./export_txt/prices.txt', prices);
write_gams_param_ii('./export_txt/op_link_cost.txt', op_link_cost);
write_gams_param_ii('./export_txt/candidates.txt', candidates);
write_gams_param_ii('./export_txt/congestion_coefs_links.txt', congestion_coef_links);

%For BLO
write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
write_gams_param_ii('./export_txt/beta_od.txt', beta_od);

fid = fopen("./export_txt/gamma.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', gamma);
fclose(fid);


% === 1D vectores ===
write_gams_param1d_full('./export_txt/station_cost.txt', station_cost);
write_gams_param1d_full('./export_txt/hub_cost.txt', hub_cost);
write_gams_param1d_full('./export_txt/station_capacity_slope.txt', station_capacity_slope);
write_gams_param1d_full('./export_txt/congestion_coefs_stations.txt', congestion_coef_stations);

a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);
sh_prev = 1e-3*s_prev;

write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);


%% Parameter definition

alfa = 0.5;
budgets = [3e4,3.5e4,4e4,4.5e4,5e4];

budgets = [3e4,3.5e4,4e4,4.5e4,5e4,5.5e4,6e4,7e4,8e4];
budgets = 3e4;
%budgets = [4e4];
%budgets = 5e4;
%budgets = [4e4,5e4,6e4,7e4,8e4,9e4];


lam = 4;
%% run MIP

for bb=1:length(budgets)
    bud = budgets(bb);
    [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,alfa,n,bud);
end
%%

for bb=1:length(budgets)
    bud = budgets(bb);
    filename = sprintf('./8node_hs_prueba_v0/bud=%d_lam=%d',bud,lam);
    load(filename);
    disp(bud)
    disp(obj_val);
   % disp(used_budget);
end

%% run cvx model


alfas = 0.1;
gamma = 20;
fid = fopen("./export_txt/gamma.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', gamma);
fclose(fid);


for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);

        % Initialize alfa_od and beta_od, gamma
        alfa_od = ones(n);
        beta_od = ones(n);
        % beta_od(3,4) = 1.3;
        % beta_od(4,3) = beta_od(3,4);
        % beta_od(2,6) = 0.7;
        % beta_od(6,2) = 0.7;

        %For BLO
        write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
        write_gams_param_ii('./export_txt/beta_od.txt', beta_od);

        [s,sh,a,f,fext,fij] = compute_sim_cvx_blo(lam,alfa,n,bud);
    end
end


% check solutions quality 

best_obj = 1e3*ones(1,length(budgets));
best_alfa = zeros(1,length(budgets));
used_bud = best_alfa;

%%

for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        filename = sprintf('./8node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d',bud,lam,alfa);
        load(filename);
        obj_val_ll
        used_bud
        if obj_val_ll < best_obj(bb)
                best_obj(bb) = obj_val_ll;
                best_alfa(bb) = alfa;
                used_bud(bb) = used_budget;
        end
    end
end
%% dibujar la red





%%


close all;
figure('Position', [100, 100, 1000, 800]);

azul_col = [0 0.4470 0.7410]; %icvx
rojo_col = [0.8500 0.3250 0.0980]; %mip30
naranja_col = [0.9290 0.6940 0.1250]; %MIP10
verde_col = [0.4660 0.6740 0.1880]; %mipreg

subplot(221);
% A y B: nObs x nX  (cada columna = un valor de X)
[nObs, nX] = size(dif');

posA = (1:nX) - 0.15;   % desplazamiento izquierda para A
posB = (1:nX) + 0.15;   % desplazamiento derecha para B

cla; hold on
boxplot(dif', 'positions', posA, 'Widths', 0.25, 'Colors',[0 0 0], 'Symbol','');
boxplot(dif_10', 'positions', posB, 'Widths', 0.25, 'Colors',[0.5 0.5 0.5], 'Symbol','');

xlim([0.5, nX+0.5]); xticks(1:nX)
eur =['[',char(8364),']'];
xl = xlabel(['Budget'],'interpreter','latex');
yl = ylabel('Diff [\%]','Interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'XTick', 1:8, 'XTickLabel', budgets_aval);

% leyenda “dummy”
plot(nan,nan,'Color',[0 0 0],'LineWidth',2); plot(nan,nan,'Color',[0.5 0.5 0.5],'LineWidth',2);
legend({'MIPREG w.r.t. MIP 30 min','MIPREG w.r.t. MIP 10 min'},'Location','best','Interpreter','latex','FontSize',9); hold off
xlim([0.7 8.3]);


subplot(222);
x = 1:size(times_MIP,1);

% Estadísticos
mu_MIP = mean(times_MIP,2);
sigma_MIP = std(times_MIP,0,2);


mu_MIP_10 = mean(times_MIP_10,2);
sigma_MIP_10 = std(times_MIP_10,0,2);

mu_MIPREG = mean(times_MIPREG,2);
sigma_MIPREG = std(times_MIPREG,0,2);

% Curvas de +1 y -1 desviación
upper_MIP = mu_MIP + sigma_MIP;
lower_MIP = mu_MIP - sigma_MIP;

upper_MIP_10 = mu_MIP_10 + sigma_MIP_10;
lower_MIP_10 = mu_MIP_10 - sigma_MIP_10;

upper_MIPREG = mu_MIPREG + sigma_MIPREG;
lower_MIPREG = mu_MIPREG - sigma_MIPREG;

% Área sombreada (patch o fill)
fill([x fliplr(x)], [upper_MIP' fliplr(lower_MIP')], ...
    rojo_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h1 = plot(x, mu_MIP, 's--', 'LineWidth',2,'Color',rojo_col);
hold on;

fill([x fliplr(x)], [upper_MIP_10' fliplr(lower_MIP_10')], ...
    naranja_col, 'EdgeColor','none', 'FaceAlpha',0.1);
hold on;
% Media
h2 = plot(x, mu_MIP_10, '*-.', 'LineWidth',2,'Color',naranja_col);
hold on;

fill([x fliplr(x)], [upper_MIPREG' fliplr(lower_MIPREG')], ...
    verde_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h3 = plot(x, mu_MIPREG, 'x-', 'LineWidth',2,'Color',verde_col);

grid on;
xl = xlabel(['Budget'],'interpreter','latex');
yl = ylabel('$t_{comp} [s]$','Interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'XTick', 1:8, 'XTickLabel', budgets_aval);
set(gca, 'TickLabelInterpreter', 'latex');
legend([h1 h2 h3], {'MIP 30 min','MIP 10 min','MIPREG'}, 'Interpreter','latex','Location','best','FontSize',9 );
xlim([1 8]);


subplot(223);

x = 1:size(links_MIP,1);

% Estadísticos
mu_MIP = mean(links_MIP,2);
sigma_MIP = std(links_MIP,0,2);

mu_MIP_10 = mean(links_MIP_10,2);
sigma_MIP_10 = std(links_MIP_10,0,2);

mu_MIPREG = mean(links_MIPREG,2);
sigma_MIPREG = std(links_MIPREG,0,2);

% Curvas de +1 y -1 desviación
upper_MIP = mu_MIP + sigma_MIP;
lower_MIP = mu_MIP - sigma_MIP;

upper_MIP_10 = mu_MIP_10 + sigma_MIP_10;
lower_MIP_10 = mu_MIP_10 - sigma_MIP_10;

upper_MIPREG = mu_MIPREG + sigma_MIPREG;
lower_MIPREG = mu_MIPREG - sigma_MIPREG;

% Área sombreada (patch o fill)
fill([x fliplr(x)], [upper_MIP' fliplr(lower_MIP')], ...
    rojo_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h1 = plot(x, mu_MIP, 's--', 'LineWidth',2,'Color',rojo_col);

hold on;
% Área sombreada (patch o fill)
fill([x fliplr(x)], [upper_MIP_10' fliplr(lower_MIP_10')], ...
    rojo_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h2 = plot(x, mu_MIP_10, '*-.', 'LineWidth',2,'Color',naranja_col);
hold on;

fill([x fliplr(x)], [upper_MIPREG' fliplr(lower_MIPREG')], ...
    verde_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h3 = plot(x, mu_MIPREG, 'x-', 'LineWidth',2,'Color',verde_col);
grid on;
xl = xlabel(['Budget'],'interpreter','latex');
yl = ylabel('$N_{arcs}$','Interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'XTick', 1:8, 'XTickLabel', budgets_aval);
set(gca, 'TickLabelInterpreter', 'latex');
legend([h1 h2 h3], {'MIP 30 min','MIP 10 min','MIPREG'}, 'Interpreter','latex','Location','best','FontSize',9 );
xlim([1 8]);


subplot(224);
x = 1:size(gaps,1);

% Estadísticos
mu_MIP = mean(gaps,2);
sigma_MIP = std(gaps,0,2);

mu_MIP_10 = mean(gaps_10,2);
sigma_MIP_10 = std(gaps_10,0,2);

% Curvas de +1 y -1 desviación
upper_MIP = mu_MIP + sigma_MIP;
lower_MIP = mu_MIP - sigma_MIP;

upper_MIP_10 = mu_MIP_10 + sigma_MIP_10;
lower_MIP_10 = mu_MIP_10 - sigma_MIP_10;

% Área sombreada (patch o fill)
fill([x fliplr(x)], [upper_MIP' fliplr(lower_MIP')], ...
    rojo_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h1 = plot(x, mu_MIP, 's--', 'LineWidth',2,'Color',rojo_col);

hold on;

fill([x fliplr(x)], [upper_MIP_10' fliplr(lower_MIP_10')], ...
    naranja_col, 'EdgeColor','none', 'FaceAlpha',0.1);

hold on;
% Media
h2 = plot(x, mu_MIP_10, '*-.', 'LineWidth',2,'Color',naranja_col);

grid on;
xl = xlabel(['Budget'],'interpreter','latex');
yl = ylabel('Optimality gap [\%]','Interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'XTick', 1:8, 'XTickLabel', budgets_aval);
set(gca, 'TickLabelInterpreter', 'latex');
legend([h1,h2], {'MIP 30 min','MIP 10 min'}, 'Interpreter','latex','Location','best','FontSize',9 )
xlim([1 8]);


saveas(gcf, './8node_random_results.png');

%% Functions


function budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam)
budget = 0;
for i=1:n
    if s(i) > 1e-2
        budget = budget + station_cost(i)+ ...
            station_capacity_slope(i)*(s(i) + sh(i));
    end
    if sh(i) > 1e-2
        budget = budget + lam*hub_cost(i);
    end
    for j=1:n
        if a(i,j) > 1e-2
            budget = budget + link_cost(i,j);
        end
    end
end
end

function [pax_obj] = get_entr_val(travel_time,prices,alt_time,alt_price,a_prim,delta_a,...
    s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op,n)

pax_obj = 0;
for o=1:n
    for d=1:n
        pax_obj = pax_obj + 1e-6*(demand(o,d).*sum(sum((travel_time+prices).*fij(:,:,o,d))));
    end
end
pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(alt_time+alt_price).*fext)));
pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(f) - f))));
pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(fext) - fext))));
pax_obj = 1e6.*pax_obj;

end

function [obj_val,pax_obj,op_obj] = get_obj_val(op_link_cost,...
    prices,a,f,demand)
n = size(a,1);
pax_obj = 0;
op_obj = 0;


op_obj = op_obj + (sum(sum(op_link_cost.*a))); %operational costs
for o=1:n
    for d=1:n
        pax_obj = pax_obj - prices(o,d)*demand(o,d).*f(o,d);
    end
end
obj_val = pax_obj + op_obj;
end



function [sprim,s,deltas,aprim,a,deltaa,f,fext,fij,comp_time] = compute_sim_MIP_entr(lam,beta,alfa,n,budget)

tic;
fid = fopen("./export_txt/lam.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', lam);
fclose(fid);

fid = fopen("./export_txt/alfa.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', alfa);
fclose(fid);

fid = fopen("./export_txt/beta.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', beta);
fclose(fid);


a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);
write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);

dm_pax = 0.01;
dm_op = 0.008;

fid = fopen("./export_txt/dm_pax.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', dm_pax);
fclose(fid);

fid = fopen("./export_txt/dm_op.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', dm_op);
fclose(fid);

fid = fopen("./export_txt/budget.txt",'w');
if fid < 0, error('No puedo abrir %s', filename); end
fprintf(fid, '%d', budget);
fclose(fid);

% fid = fopen("./export_txt/niters.txt",'w');
% if fid < 0, error('No puedo abrir %s', filename); end
%     fprintf(fid, '%d', niters);
% fclose(fid);


cmdpath = "cd C:\GAMS\50";

system(cmdpath);

gmsFile  = 'C:\Users\freal\MATLAB\Projects\untitled\code\8node\mipreg_mosek.gms';

gamsExe = 'C:\GAMS\50\gams.exe';



cmd = sprintf('%s %s ', ...
    gamsExe, gmsFile);
[status,out] = system(cmd);
disp(out);

results_file_ctime = readtable('./output_all.xlsx','Sheet','solver_time');
comp_time = table2array(results_file_ctime);


results_file_sprim = readtable('./output_all.xlsx','Sheet','sprim_level');
sprim = table2array(results_file_sprim(1,:));

results_file_s = readtable('./output_all.xlsx','Sheet','s_level');
s = table2array(results_file_s(1,:));

results_file_deltas = readtable('./output_all.xlsx','Sheet','deltas_level');
deltas = table2array(results_file_deltas(1,:));

results_file_aprim = readtable('./output_all.xlsx','Sheet','aprim_level');
aprim = table2array(results_file_aprim(1:n,2:(n+1)));

results_file_a = readtable('./output_all.xlsx','Sheet','a_level');
a = table2array(results_file_a(1:n,2:(n+1)));

results_file_deltaa = readtable('./output_all.xlsx','Sheet','deltaa_level');
deltaa = table2array(results_file_deltaa(1:n,2:(n+1)));

results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
f = table2array(results_file_f(1:n,2:(n+1)));

results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
fext = table2array(results_file_fext(1:n,2:(n+1)));

T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
[iU,~,iIdx] = unique(T.i,'stable');
[jU,~,jIdx] = unique(T.j,'stable');
[oU,~,oIdx] = unique(T.o,'stable');
[dU,~,dIdx] = unique(T.d,'stable');


fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
    [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);

a_bin = zeros(n);
a_bin(aprim > 1e-2) = 1;

s_bin = zeros(n,1);
s_bin(sprim > 1e-2) = 1;

write_gams_param_ii('./export_txt/a_bin_mipreg.txt', a_bin);
write_gams_param_ii('./export_txt/a_prim_mipreg.txt', max(0,aprim));
write_gams_param1d_full('./export_txt/s_bin_mipreg.txt', s_bin);
write_gams_param1d_full('./export_txt/s_prim_mipreg.txt', max(0,sprim));

gmsFile  = 'C:\Users\freal\MATLAB\Projects\untitled\code\8node\flow_assignment.gms';

gamsExe = 'C:\GAMS\50\gams.exe';



cmd = sprintf('%s %s ', ...
    gamsExe, gmsFile);
[status,out] = system(cmd);
disp(out);


results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
f = table2array(results_file_f(1:n,2:(n+1)));

results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
fext = table2array(results_file_fext(1:n,2:(n+1)));

T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
[iU,~,iIdx] = unique(T.i,'stable');
[jU,~,jIdx] = unique(T.j,'stable');
[oU,~,oIdx] = unique(T.o,'stable');
[dU,~,dIdx] = unique(T.d,'stable');


fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
    [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);



end


function [s,sh,a,f,fext,fij] = compute_sim_cvx_blo(lam,alfa,n,budget)


[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_8node_network();



niters = 10;

mu_alfa = 1e-7; mu_beta = 1e-1;

alfa_od = ones(n);
beta_od = ones(n);

write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
% beta_od(3,4) = 1.3;
% beta_od(4,3) = beta_od(3,4);
% beta_od(2,6) = 0.7;
% beta_od(6,2) = 0.7;
gamma = 20;

logit_coef = 0.02; n_airlines = 5;

fid = fopen("./export_txt/niters.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', niters);
fclose(fid);

fid = fopen("./export_txt/lam.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', lam);
fclose(fid);

fid = fopen("./export_txt/alfa.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', alfa);
fclose(fid);

fid = fopen("./export_txt/budget.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', budget);
fclose(fid);
disp('este es el presupuesto:')
disp(budget);

obj_hist = zeros(1,30);   % histórico
figure;
h = plot(nan, nan, '-o', 'LineWidth', 2);
grid on;
xlabel('Iteración BL');
ylabel('obj\_val\_ll');
title('Evolución del valor objetivo');
bliters = 30;

a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);

sh_prev = s_prev;

comp_time = 0;

obj_val = 0;
obj_val_prev = 1e3;

for iter=1:niters
    % alfa_od = ones(n);
    % beta_od = ones(n);
    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
    write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
    stop = 0;
    for bliter=1:bliters
    
        if (abs((obj_val-obj_val_prev)./obj_val) <= 1e-3) && (bliter > 1)
            stop = 1;
        elseif (stop == 0)
        
    
    
            write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
            write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
            write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);
            
            
            %cmdpath = "cd C:\GAMS\50";
            cmdpath = "cd C:\GAMS\50";
            
            system(cmdpath);
            gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-ll.gms';
            
            gamsExe = 'C:\GAMS\50\gams.exe'; 
            
            
            cmd = sprintf('%s %s ', ...
                gamsExe, gmsFile);
            iter = 1;
            
    
            fid = fopen("./export_txt/current_iter.txt",'w');
            if fid < 0, error('No puedo abrir %s', fid); end
            fprintf(fid, '%d', iter);
            fclose(fid);
            [status,out] = system(cmd);
            % disp(out);
        
            results_file_ctime = readtable('./output_all.xlsx','Sheet','solver_time');
            comp_time = comp_time + table2array(results_file_ctime);
        
            results_file_sh = readtable('./output_all.xlsx','Sheet','sh_level');
            sh = table2array(results_file_sh(1,:));
            sh = max(sh,1e-4);
       
    
    
        
        
            results_file_s = readtable('./output_all.xlsx','Sheet','s_level');
            s = table2array(results_file_s(1,:));
            s = max(s,1e-4);
         
            s
            sh
    
            results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
            f = table2array(results_file_f(1:n,2:(n+1)));
            f
    
    
    
        
            results_file_a = readtable('./output_all.xlsx','Sheet','a_level');
            a = table2array(results_file_a(1:n,2:(n+1)));
            a = max(a,1e-4);
    
            % 
            % write_gams_param_ii('./export_txt/a_prev.txt', a);
            % write_gams_param1d_full('./export_txt/s_prev.txt', s);
            % write_gams_param1d_full('./export_txt/sh_prev.txt', sh);
        
            
            
            results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
            f = table2array(results_file_f(1:n,2:(n+1)));
            
            
            results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
            fext = table2array(results_file_fext(1:n,2:(n+1)));
            
            
            T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
            [iU,~,iIdx] = unique(T.i,'stable');
            [jU,~,jIdx] = unique(T.j,'stable');
            [oU,~,oIdx] = unique(T.o,'stable');
            [dU,~,dIdx] = unique(T.d,'stable');
            
            
            fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
                [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);
            
            a(a < 1e-2) = 0;
            f(f < 1e-2) = 0;
            fij(fij < 1e-2) = 0;
            fext(fext > 0.99) = 1;
    
            
            a_ll = a;
            f_ll = f;

            s_ll = s;
            sh_ll = sh;
            

            
            %comprobar que está bien
            grad_alfa_v = zeros(n);
            grad_beta_v = zeros(n);
            for oo=1:n
                for dd=1:n
                    
                    propio = ((prices(oo,dd) * demand(oo,dd)).^beta_od(oo,dd)) * (logit_coef*prices(oo,dd)*f(oo,dd) + logit_coef*sum(sum( squeeze(fij(:,:,oo,dd)).*travel_time ))   );
                    externo = - ((prices(oo,dd) * demand(oo,dd)).^beta_od(oo,dd)) * (alt_utility(oo,dd)*fext(oo,dd) );
                    log_propio = ((prices(oo,dd) * demand(oo,dd)).^beta_od(oo,dd)) * (f(oo,dd)*( log(max(0,f(oo,dd)) + 1e-12) - 1 ));
                    log_ext = ((prices(oo,dd) * demand(oo,dd)).^beta_od(oo,dd)) * (fext(oo,dd)*( log(max(0,fext(oo,dd))/n_airlines + 1e-12)  - 1 ));
        
        
                    grad_alfa_v(oo,dd) = propio+externo+log_propio+log_ext;
                    grad_beta_v(oo,dd) = (alfa_od(oo,dd)+1e-4)*((beta_od(oo,dd)*demand(oo,dd)*prices(oo,dd)).^(beta_od(oo,dd)-1)) * (logit_coef * prices(oo,dd) * f(oo,dd) + logit_coef * sum(sum(squeeze(fij(:,:,oo,dd)) .* travel_time)) - alt_utility(oo,dd) * fext(oo,dd) + f(oo,dd)*( log(max(0,f(oo,dd)) + 1e-12) - 1 ) + fext(oo,dd)*( log(max(0,fext(oo,dd))/n_airlines + 1e-12)  - 1 ) );
                end
            end

            used_budget = get_budget(s,sh,a,n,...
                station_cost,station_capacity_slope,hub_cost,link_cost,lam)

            
            
            
            
            % a_prev = 1e3*ones(n);
            % s_prev = 1e4*ones(1,n);
            % 
            % sh_prev = 1e-2*s_prev;
            % 
            % write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
            % write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
            % write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);
    
            set_max_f(n,fij,n_airlines,travel_time,prices,alt_utility,omega_p,omega_t);
               
            
            gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-sl.gms';
            
            gamsExe = 'C:\GAMS\50\gams.exe'; 
            
            
            cmd = sprintf('%s %s ', ...
                gamsExe, gmsFile);
            
            
    
            % fid = fopen("./export_txt/current_iter.txt",'w');
            % if fid < 0, error('No puedo abrir %s', fid); end
            % fprintf(fid, '%d', iter);
            % fclose(fid);
            [status,out] = system(cmd);
            %disp(out);
        
            results_file_ctime = readtable('./output_all.xlsx','Sheet','solver_time');
            comp_time = comp_time + table2array(results_file_ctime);
        
            results_file_sh = readtable('./output_all.xlsx','Sheet','sh_level');
            sh = table2array(results_file_sh(1,:));
            sh = max(sh,1e-4);
        
        
            results_file_s = readtable('./output_all.xlsx','Sheet','s_level');
            s = table2array(results_file_s(1,:));
            s = max(s,1e-4);

        
            results_file_a = readtable('./output_all.xlsx','Sheet','a_level');
            a = table2array(results_file_a(1:n,2:(n+1)));
            a = max(a,1e-4);
            
                % write_gams_param_ii('./export_txt/a_prev.txt', a);
                % write_gams_param1d_full('./export_txt/s_prev.txt', s);
                % write_gams_param1d_full('./export_txt/sh_prev.txt', sh);
            
            
    
            
            results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
            f = table2array(results_file_f(1:n,2:(n+1)));
            
            results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
            fext = table2array(results_file_fext(1:n,2:(n+1)));
            
            
            T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
            [iU,~,iIdx] = unique(T.i,'stable');
            [jU,~,jIdx] = unique(T.j,'stable');
            [oU,~,oIdx] = unique(T.o,'stable');
            [dU,~,dIdx] = unique(T.d,'stable');
            
            
            fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
                [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);
            
            a(a < 1e-2) = 0;
            f(f < 1e-2) = 0;
            fext(fext > 0.99) = 1;
            fij(fij < 1e-2) = 0;
            
            %disp('f del sl:');
            %disp(f);
            
            
            
            
            [obj_val_ll,pax_obj,op_obj] = get_obj_val(op_link_cost,...
            prices,a_ll,f_ll,demand);
            used_budget = get_budget(s,sh,a,n,...
                station_cost,station_capacity_slope,hub_cost,link_cost,lam);
            
            disp(obj_val_ll);
            
            f_sl = f;
            a_sl = a;
        
        
            grad_alfa_f = zeros(n);
            grad_beta_f = zeros(n);
            for oo=1:n
                for dd=1:n
                    grad_alfa_f(oo,dd) = gamma .*( ((demand(oo,dd)*prices(oo,dd)).^(beta_od(oo,dd))) * (logit_coef * prices(oo,dd) * f(oo,dd) + logit_coef * sum(sum(squeeze(fij(:,:,oo,dd)) .* travel_time)) - alt_utility(oo,dd) * fext(oo,dd) + f(oo,dd)*( log(f(oo,dd) + 1e-12) - 1 ) + fext(oo,dd)*( log(fext(oo,dd)/n_airlines + 1e-12)  - 1 ) ) - grad_alfa_v(oo,dd));
                    grad_beta_f(oo,dd) = gamma .*( (alfa_od(oo,dd)+1e-4)*((beta_od(oo,dd)*demand(oo,dd)*prices(oo,dd)).^(beta_od(oo,dd)-1)) * (logit_coef * prices(oo,dd) * f(oo,dd) + logit_coef * sum(sum(squeeze(fij(:,:,oo,dd)) .* travel_time)) - alt_utility(oo,dd) * fext(oo,dd) + f(oo,dd)*( log(f(oo,dd) + 1e-12) - 1 ) + fext(oo,dd)*( log(fext(oo,dd)/n_airlines + 1e-12)  - 1 ) ) - grad_beta_v(oo,dd));
                end
            end
            
        
            beta_od = beta_od - mu_beta.*grad_beta_f;
            alfa_od = alfa_od - mu_alfa.*grad_alfa_f;
        
            beta_od = max(0.5,beta_od);
            beta_od = min(2.1,beta_od);
            
        
            alfa_od = max(1,alfa_od);
            alfa_od = min(9,alfa_od);
        
            write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
            write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
            
            disp('beta');
            disp(beta_od);
            disp('alfa');
            disp(alfa_od);
            disp('obj_val');
            disp(obj_val_ll);
        
            obj_hist(bliter) = obj_val_ll;
        
            set(h, 'XData', 1:bliter, ...
                   'YData', obj_hist(1:bliter));
            
            drawnow;
            obj_val_prev = obj_val;
            obj_val = obj_val_ll;
        end
    
    end
    if ((used_budget - budget)/budget) < 0.05
        disp('cumplo presupuesto');
        break;
    end
    s_prev = s_ll;
    sh_prev = sh_ll;
    a_prev = a_ll;
    stop = 0;

end

a_prev = 1e4*ones(n);
s_prev = 0.1*ones(1,n);
sh_prev = s_prev;

cmdpath = "cd C:\GAMS\50";

system(cmdpath);
gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-ll.gms';

gamsExe = 'C:\GAMS\50\gams.exe'; 


cmd = sprintf('%s %s ', ...
    gamsExe, gmsFile);

for iter=1:niters

        write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
        write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
        write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);


        %cmdpath = "cd C:\GAMS\50";
        cmdpath = "cd C:\GAMS\50";

        system(cmdpath);
        gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-mip.gms';

        gamsExe = 'C:\GAMS\50\gams.exe'; 


        cmd = sprintf('%s %s ', ...
            gamsExe, gmsFile);


        fid = fopen("./export_txt/current_iter.txt",'w');
        if fid < 0, error('No puedo abrir %s', fid); end
        fprintf(fid, '%d', iter);
        fclose(fid);
        [status,out] = system(cmd);
        % disp(out);

        results_file_ctime = readtable('./output_all.xlsx','Sheet','solver_time');
        comp_time = comp_time + table2array(results_file_ctime);

        results_file_sh = readtable('./output_all.xlsx','Sheet','sh_level');
        sh = table2array(results_file_sh(1,:));
        sh = max(sh,1e-4);





        results_file_s = readtable('./output_all.xlsx','Sheet','s_level');
        s = table2array(results_file_s(1,:));
        s = max(s,1e-4);

        s
        sh

        results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
        f = table2array(results_file_f(1:n,2:(n+1)));
        f




        results_file_a = readtable('./output_all.xlsx','Sheet','a_level');
        a = table2array(results_file_a(1:n,2:(n+1)));
        a = max(a,1e-4);

        % 
        % write_gams_param_ii('./export_txt/a_prev.txt', a);
        % write_gams_param1d_full('./export_txt/s_prev.txt', s);
        % write_gams_param1d_full('./export_txt/sh_prev.txt', sh);



        results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
        f = table2array(results_file_f(1:n,2:(n+1)));


        results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
        fext = table2array(results_file_fext(1:n,2:(n+1)));


        T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
        [iU,~,iIdx] = unique(T.i,'stable');
        [jU,~,jIdx] = unique(T.j,'stable');
        [oU,~,oIdx] = unique(T.o,'stable');
        [dU,~,dIdx] = unique(T.d,'stable');


        fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
            [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);

        a(a < 1e-2) = 0;
        f(f < 1e-2) = 0;
        fij(fij < 1e-2) = 0;
        fext(fext > 0.99) = 1;
        a_prev = a;
        s_prev = s;
        sh_prev = sh;
end

s = s_ll; sh = sh_ll; a = a_ll; f = f_ll;

filename = sprintf('./8node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d.mat',budget,lam,alfa);
save(filename,'s','sh', ...
    'a','f','fext','fij','comp_time','used_budget', ...
    'pax_obj','op_obj','obj_val_ll','alfa_od','beta_od','obj_hist');

end




function [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,beta,alfa,n,budget)

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidasourcertes] = parameters_8node_network();

tic;

dm_pax = 0.01;
dm_op = 0.008;

fid = fopen("./export_txt/lam.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', lam);
fclose(fid);

fid = fopen("./export_txt/alfa.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', alfa);
fclose(fid);

fid = fopen("./export_txt/beta.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', beta);
fclose(fid);

fid = fopen("./export_txt/budget.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', budget);
fclose(fid);

disp(budget);

a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);
write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);


fid = fopen("./export_txt/dm_pax.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', dm_pax);
fclose(fid);

fid = fopen("./export_txt/dm_op.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', dm_op);
fclose(fid);




%cmdpath = "cd C:\GAMS\50";
cmdpath = "cd /Library/Frameworks/GAMS.framework/Resources"

system(cmdpath);

gmsFile  = '/Users/fernandorealrojas/Desktop/HubSpokeNetworkDesign/8node_spain/mip.gms';

gamsExe = '/Library/Frameworks/GAMS.framework/Resources/gams';



cmd = sprintf('%s %s ', ...
    gamsExe, gmsFile);
[status,out] = system(cmd);
disp(out);

results_file_sh = readtable('./output_all.xlsx','Sheet','sh_level');
sh = table2array(results_file_sh(1,:));

results_file_s = readtable('./output_all.xlsx','Sheet','s_level');
s = table2array(results_file_s(1,:));
sprim = s;

%results_file_deltas = readtable('./output_all.xlsx','Sheet','deltas_level');
%deltas = table2array(results_file_deltas(1,:));
deltas = zeros(1,n);

%results_file_aprim = readtable('./output_all.xlsx','Sheet','aprim_level');
%aprim = table2array(results_file_aprim(1:n,2:(n+1)));

results_file_a = readtable('./output_all.xlsx','Sheet','a_level');
a = table2array(results_file_a(1:n,2:(n+1)));

%results_file_deltaa = readtable('./output_all.xlsx','Sheet','deltaa_level');
%deltaa = table2array(results_file_deltaa(1:n,2:(n+1)));

results_file_f = readtable('./output_all.xlsx','Sheet','f_level');
f = table2array(results_file_f(1:n,2:(n+1)));

results_file_fext = readtable('./output_all.xlsx','Sheet','fext_level');
fext = table2array(results_file_fext(1:n,2:(n+1)));

results_file_mipgap = readtable('./output_all.xlsx','Sheet','mip_opt_gap');
mipgap = table2array(results_file_mipgap);

T = readtable('fij_long.csv');      % columnas: i, j, o, d, value (strings/números)
[iU,~,iIdx] = unique(T.i,'stable');
[jU,~,jIdx] = unique(T.j,'stable');
[oU,~,oIdx] = unique(T.o,'stable');
[dU,~,dIdx] = unique(T.d,'stable');


fij = accumarray([iIdx,jIdx,oIdx,dIdx], T.value, ...
    [numel(iU), numel(jU), numel(oU), numel(dU)], @sum, 0);

results_file_ctime = readtable('./output_all.xlsx','Sheet','solver_time');
comp_time = table2array(results_file_ctime);
[obj_val,pax_obj,op_obj] = get_obj_val(op_link_cost,...
    prices,a,f,demand)
budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam);
filename = sprintf('./8node_hs_prueba_v0/beta=%d_lam=%d.mat',beta,lam);
save(filename,'s','sprim','deltas', ...
    'a','f','fext','fij','comp_time','budget', ...
    'pax_obj','op_obj','obj_val','mipgap');

end

% compute_sim_MIP_rand removed



function [n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_8node_network()

n = 8;

%candidates to construct a link for each neighbor
candidates = {[2,3],[1,3,4],[1,2,4,5],[2,3,5,6],[3,4,6],[4,5]};

%cost of using the alternative network for each o-d pair
alt_cost = [0,1.6,0.8,2,1.6,2.5;
    2,0,0.9,1.2,1.5,2.5;
    1.5,1.4,0,1.3,0.9,2;
    1.9,2,1.9,0,1.8,2;
    3,1.5,2,2,0,1.5;
    2.1,2.7,2.2,1,1.5,0];

%fixed cost for constructing links
link_cost = (1e6/(25*365.25)).*[0,1.7,2.7,0,0,0;
    1.7,0,2.1,3,0,0;
    2.7,2.1,0,2.6,1.7,0;
    0,3,2.6,0,2.8,2.4;
    0,0,1.7,2.8,0,1.9;
    0,0,0,2.4,1.9,0];
link_cost (link_cost ==0) = 1e4;


%fixed cost for constructing stations
station_cost = (1e6/(25*365.25)).*[2, 3, 2.2, 3, 2.5, 1.3];

hub_cost = (1e6/(25*365.25)).*[200, 100, 2, 300, 200, 100];

link_capacity_slope = 0.04.*link_cost;
station_capacity_slope = 0.04.*station_cost;

%demand between od pairs
demand = 1e3.*[0,9,26,19,13,12;
    11,0,14,26,7,18;
    30,19,0,30,24,8;
    21,9,11,0,22,16;
    14,14,8,9,0,20;
    26,1,22,24,13,0];

distance = 1e4 * ones(n); % Distances between arcs

for i = 1:n
    distance(i, i) = 0;
end

distance(1, 2) = 0.75;
distance(1, 3) = 0.7;


distance(2, 3) = 0.6;
distance(2, 4) = 1.1;

distance(3, 4) = 1.1;
distance(3, 5) = 0.5;


distance(4,5) = 0.8;
distance(4,6) = 0.7;


distance(5,6) = 0.5;


for i = 1:n
    for j = i+1:n
        distance(j, i) = distance(i, j); % Distances are symmetric
    end
end

%Load factor on stations
load_factor = 0.25 .* ones(1, n);

% Op Link Cost
op_link_cost = 4.*distance;

% Congestion Coefficients
congestion_coef_stations = 0.1 .* ones(1, n);
congestion_coef_links = 0.1 .* ones(n);

% Prices
prices = (distance).^(0.7);
prices(prices > 10) = 1.2;
%prices = zeros(n);

% Travel Time
travel_time = 60 .* distance ./ 30; % Time in minutes

% Alt Time
alt_time = 60 .* alt_cost ./ 30; % Time in minutes

alt_price = (alt_cost).^(0.7); %price

a_nom = 588;

tau = 0.57;
eta = 0.25;
a_max = 1e9;
eps = 1e-3;
end

% parameters_8node_network_rand removed


function [lin_coef,bord,b] = get_linearization(n,nreg,alt_utility,vals_regs,n_airlines)
dmax = zeros(nreg,n,n);
dmin = dmax;
lin_coef = dmax;
bord = zeros(nreg,n,n);


for o=1:n
    for d=1:n
        u = alt_utility(o,d);
        for r=1:(nreg-1)
            dmax(r,o,d) = min(0,u + log(n_airlines*vals_regs(r)/(1-vals_regs(r)))  );
        end
        dmax(nreg,o,d) = 0;
        dmin(1,o,d) = -3e1;
        for r=2:nreg
            dmin(r,o,d) = dmax(r-1,o,d);
        end
        
        for r=2:(nreg-1)
            if (dmax(r,o,d) == dmin(r,o,d))
                lin_coef(r,o,d) = 0;
                bord(r,o,d) = vals_regs(r);
            else
                lin_coef(r,o,d) = (vals_regs(r)-vals_regs(r-1))/(dmax(r,o,d)-dmin(r,o,d));
                bord(r,o,d) = vals_regs(r-1);
            end
        end
        lin_coef(1,o,d) = (vals_regs(1))/(dmax(1,o,d)-dmin(1,o,d));
        bord(1,o,d) = 0;
        if dmin(nreg,o,d)==0
            lin_coef(nreg,o,d) = 0;
        else
            lin_coef(nreg,o,d) = (1-vals_regs(nreg-1))/(0-dmin(nreg,o,d));
        end
        bord(nreg,o,d) = vals_regs(nreg-1);
    end
end
b = dmin;


end

function val = logit(x,omega_t,omega_p,time,price)
val = exp(x)./( exp(x) + exp(omega_t*time + omega_p*price) );
end

function write_gams_param_iii(filename, M)
% filename: ruta del .txt (p.ej. 'demand.txt')
% M: matriz NxN (puede ser sparse)
% zero_tol: umbral para considerar cero (p.ej. 0 o 1e-12)

[n1, n2, n3] = size(M);


fid = fopen(filename,'w');
if fid < 0, error('No puedo abrir %s', filename); end

% Si M es dispersa, recorre solo no-ceros

for s=1:n1
    for r = 1:n2
        for c = 1:n3
            val = M(s,r,c);
            fprintf(fid, 'seg%d.i%d.i%d %.12g\n', s, r, c, val);
        end
    end
end
fclose(fid);
end


function write_gams_param_ii(filename, M)
% filename: ruta del .txt (p.ej. 'demand.txt')
% M: matriz NxN (puede ser sparse)
% zero_tol: umbral para considerar cero (p.ej. 0 o 1e-12)

if nargin < 3; end
[n1, n2] = size(M);
if n1 ~= n2
    error('M debe ser cuadrada para dominio (i,i).');
end

fid = fopen(filename,'w');
if fid < 0, error('No puedo abrir %s', filename); end

% Si M es dispersa, recorre solo no-ceros
if issparse(M)
    [r,c,v] = find(M);
    for k = 1:numel(v)
        fprintf(fid, 'i%d.i%d %.12g\n', r(k), c(k), v(k));
    end
else
    for r = 1:n1
        for c = 1:n2
            val = M(r,c);
            fprintf(fid, 'i%d.i%d %.12g\n', r, c, val);
        end
    end
end
fclose(fid);
end

function write_gams_param1d_full(filename, v)
% Escribe un parámetro 1D en formato GAMS:
%   Parameter <paramName> /
%   i1 <valor>
%   i2 <valor>
%   ...
%   /;
%
% filename : ruta del .txt (p.ej. 'station_cost.txt')
% v        : vector (Nx1 o 1xN)
% paramName: nombre del parámetro en GAMS (def: 'station_cost')
% prefix   : prefijo de la etiqueta (def: 'i' -> i1, i2, ...)
% zero_tol : umbral para omitir ~0 (def: 0 => no escribe los ceros exactos)
v = v(:);
n = numel(v);

fid = fopen(filename,'w');
if fid < 0, error('No puedo abrir %s', filename); end

for k = 1:n
    val = v(k);
    fprintf(fid, 'i%d %.12g\n', k, val);
end
fclose(fid);
end

function [n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidates,omega_t,omega_p] = parameters_8node_network()
% 8 aeropuertos más usados de España: MAD, BCN, PMI, AGP, ALC, LPA, TFS, IBZ
% Datos reales leídos desde CSVs generados por extract_data.py y extract_demand_8node.py

n = 8;
n_airlines = 5;

% Candidates: Full connectivity assumption
candidates = ones(n) - eye(n);

% Distances (km) - generadas por extract_data.py
distance = readmatrix('distance.csv');

% Prices (Euros) - generadas por extract_data.py
prices = readmatrix('prices.csv');

omega_t = -0.02;
omega_p = -0.02;

% Link Cost: proporcional a la distancia (como en 8node)
link_cost = 10.* distance;
link_cost(logical(eye(n))) = 1e4; % Alto coste para self-loops
link_cost(link_cost == 0) = 1e4; % Alto coste para conexiones sin datos

station_cost = 3e3.*ones(1,n); % Oficinas, tasas del aeropuerto
hub_cost = 5e3.*ones(1,n);

link_capacity_slope = 0.2 .* link_cost;
station_capacity_slope = (5*5e2 + 4*50*8).*ones(1,n); % 500 e/dia estacionamiento + 50e/hora personal

% Demand: datos reales de pasajeros anuales - generados por extract_demand_8node.py
demand = readmatrix('demand.csv');

load_factor = 0.25 .* ones(1, n);

congestion_coef_stations = 0.1 .* ones(1, n);
congestion_coef_links = 0.1 .* ones(n);
takeoff_time = 20;
landing_time = 20;
taxi_time = 10;
cruise_time = 60 .* distance ./ 800;

travel_time = cruise_time + takeoff_time + landing_time + taxi_time; % Velocidad crucero ~800 km/h -> minutos
travel_time(logical(eye(n))) = 0;

rng(123);
p_escala = 0.4;

alt_utility = zeros(n);

for i = 1:n
    for j = i+1:n   % SOLO mitad superior

        escala = rand(1, n_airlines) < p_escala;

        % --- TIEMPO alternativa ---
        alt_time_vec = travel_time(i,j) .* (1 + 0.5 .* escala) ...
                       + 60 .* escala;

        % --- PRECIO alternativa ---
        alt_price_vec = prices(i,j) ...
            + 0.3 .* prices(i,j) .* (rand(1, n_airlines) - 0.5);

        alt_u = log(sum(exp(omega_p.*alt_price_vec + omega_t.*alt_time_vec))) - log(n_airlines);

        % Asignación simétrica
        alt_utility(i,j) = alt_u;
        alt_utility(j,i) = alt_u;
    end
end
alt_utility(1:n+1:end) = 0;

op_link_cost = 7600 .* travel_time./60; % Coste operacional proporcional al tiempo de vuelo

a_nom = 171;
tau = 0.85;
eta = 0.3;
a_max = 1e9;

end

