close all;
clear all;
clc;

% === 1) RUTA de tu GAMS ===
gamsHome = 'C:\GAMS\50';  % <-- CAMBIA esto  

[basedir,~,~] = fileparts(mfilename('fullpath'));
basedir = fullfile(basedir, 'export_csv');   % subcarpeta de exportación
if ~exist(basedir,'dir'); mkdir(basedir); end

% === Tus datos ===

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_4node_network();

%demand = demand./365;

M = 1e4;
nreg = 20;
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

%sh_prev = [5,0.5,5,5,5,5];

write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);


%% Parameter definition

budgets = [3e4,3.5e4,4e4,4.5e4,5e4];
lam = 4;
%% run MIP

for bb=1:length(budgets)
    bud = budgets(bb);
    [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,bud);
end
%%

for bb=1:length(budgets)
    bud = budgets(bb);
    filename = sprintf('./4node_hs_prueba_v0/2h_budget=%d_lam=%d',bud,lam);
    load(filename);
    disp(bud)
    disp(obj_val);
   % disp(used_budget);
end

%% define cvx model


alfas = 0.1;
gamma = 20;
fid = fopen("./export_txt/gamma.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', gamma);
fclose(fid);


mus_alfa = 1e-5; mus_beta = 5e-2;


%% run cvx blo single start


for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        for ma=1:length(mus_alfa)
            mu_alfa = mus_alfa(ma);
            for mb=1:length(mus_beta)
                mu_beta = mus_beta(mb);

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

                % Multistart en sh_prev
                sh_prev = 5 * ones(1, n);    
                %sh_prev = [5,5,5,5];

                [s,sh,a,f,fext,fij, ...
                 comp_time,used_budget,pax_obj,op_obj, ...
                 obj_val_ll,alfa_od,beta_od,obj_hist] = ...
                    compute_sim_cvx_blo(lam,alfa,n,bud,mu_alfa,mu_beta,sh_prev);
                
                obj = sum(sum(f.*demand.*prices)) - sum(sum(op_link_cost.*a));


                filename = sprintf('./4node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d_mu_al=%d_mu_bet=%d_replica8node.mat',bud,lam,alfa,mu_alfa,mu_beta);
                save(filename,'s','sh', ...
                    'a','f','fext','fij','comp_time','used_budget', ...
                    'pax_obj','op_obj','obj_val_ll','alfa_od','beta_od','obj_hist');
            end
        end
    end
end


%% run cvx blo multistart

for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        for ma=1:length(mus_alfa)
            mu_alfa = mus_alfa(ma);
            for mb=1:length(mus_beta)
                mu_beta = mus_beta(mb);

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

                % Multistart en sh_prev
                sh_prev_list = 5 * ones(n+1, n);
                for idx_ms = 1:n
                    sh_prev_list(idx_ms+1, idx_ms) = 2;
                end
                
                best_obj_ms = 1;
                best_res_ms = struct();

                best_res_ms.f = zeros(n);
                best_res_ms.a = zeros(n);

                for start_idx = 1:(n+1)
                    sh_prev_in = sh_prev_list(start_idx, :);
                    fprintf('empieazo multistart con sh = \n');
                    sh_prev_in
                    [s_curr,sh_curr,a_curr,f_curr,fext_curr,fij_curr, ...
                     comp_time_curr,used_budget_curr,pax_obj_curr,op_obj_curr, ...
                     obj_val_ll_curr,alfa_od_curr,beta_od_curr,obj_hist_curr] = ...
                        compute_sim_cvx_blo(lam,alfa,n,bud,mu_alfa,mu_beta,sh_prev_in);
                    

                    f_curr( abs(f_curr - best_res_ms.f) < 2e-2 ) = best_res_ms.f( abs(f_curr - best_res_ms.f) < 2e-2 );
                    f_curr
                    a_curr( abs(a_curr - best_res_ms.a) < 5e-2 ) = best_res_ms.a( abs(a_curr - best_res_ms.a) < 5e-2 );
                    a_curr
                    obj_curr = sum(sum(f_curr.*demand.*prices)) - sum(sum(op_link_cost.*a_curr))
       

                    if (obj_curr > best_obj_ms) && ( sum(s_curr) > 2e-2 )
                        disp('el mejor hasta ahora');
                        disp('f sale:');
                        f_curr
                        best_obj_ms = obj_curr;
                        best_res_ms.s = s_curr; best_res_ms.sh = sh_curr; best_res_ms.a = a_curr;
                        best_res_ms.f = f_curr; best_res_ms.fext = fext_curr; best_res_ms.fij = fij_curr;
                        best_res_ms.comp_time = comp_time_curr; best_res_ms.used_budget = used_budget_curr;
                        best_res_ms.pax_obj = pax_obj_curr; best_res_ms.op_obj = op_obj_curr;
                        best_res_ms.obj_val_ll = obj_val_ll_curr; best_res_ms.alfa_od = alfa_od_curr;
                        best_res_ms.beta_od = beta_od_curr; best_res_ms.obj_hist = obj_hist_curr;
                    end
                end

                % Restore the best result
                s = best_res_ms.s; sh = best_res_ms.sh; a = best_res_ms.a; f = best_res_ms.f;
                fext = best_res_ms.fext; fij = best_res_ms.fij; comp_time = best_res_ms.comp_time;
                used_budget = best_res_ms.used_budget; pax_obj = best_res_ms.pax_obj;
                op_obj = best_res_ms.op_obj; obj_val_ll = best_res_ms.obj_val_ll;
                alfa_od = best_res_ms.alfa_od; beta_od = best_res_ms.beta_od;
                obj_hist = best_res_ms.obj_hist;

                filename = sprintf('./4node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d_mu_al=%d_mu_bet=%d_replica8node.mat',bud,lam,alfa,mu_alfa,mu_beta);
                save(filename,'s','sh', ...
                    'a','f','fext','fij','comp_time','used_budget', ...
                    'pax_obj','op_obj','obj_val_ll','alfa_od','beta_od','obj_hist');
            end
        end
    end
end


% check solutions quality 

best_obj = 1e3*ones(1,length(budgets));
best_alfa = zeros(1,length(budgets));
used_bud = best_alfa;

%% load blo results


for bb=1:length(budgets)
    bud = budgets(bb);
    bud
    best_obj = 0;
    best_mu_alfa = 0;
    best_mu_beta = 0;

    filename_MIP = sprintf('./4node_hs_prueba_v0/2h_bud=%d_lam=%d',bud,lam);
    load(filename_MIP);
    f_MIP = f;
    a_MIP = a;
    
    for al=1:length(alfas)
        alfa = alfas(al);
        for ma=1:length(mus_alfa)
            mu_alfa = mus_alfa(ma);
            for mb=1:length(mus_beta)
                mu_beta = mus_beta(mb);
                filename = sprintf('./4node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d_mu_al=%d_mu_bet=%d_replica8node.mat',bud,lam,alfa,mu_alfa,mu_beta);
                load(filename);
                obj = sum(sum(f.*prices.*demand)) - sum(sum(a.*op_link_cost));
                if (obj > best_obj)
                    best_obj = obj;
                    best_f = f;
                    best_a = a;
                    best_mu_alfa = mu_alfa; best_mu_beta = mu_beta;
                end
            end
        end
    end
    f_MIP( abs(f_MIP - best_f) < 2e-2 ) = best_f( abs(f_MIP - best_f) < 2e-2 );
    a_MIP( abs(f_MIP - best_f) < 2e-2 ) = best_a( abs(f_MIP - best_f) < 2e-2 );
    % best_f
    % f_MIP
    % best_a
    % a_MIP
    obj_MIP = sum(sum(f_MIP.*prices.*demand)) - sum(sum(a_MIP.*op_link_cost));
    gap = 100.*(obj_MIP - best_obj)/obj_MIP;
    fprintf('budget = %d: best obj  = %d, mu_alfa = %d, mu_beta = %d, MIP obj = %d, gap = %d \n',bud,best_obj,best_mu_alfa,best_mu_beta,obj_MIP,gap);
end


%%

mu_beta = 2e-1;
mu_alfa = 1e-7;
budgets = [3e4,4e4,5e4,6e4,7e4,8e4,9e4,1e5];
budgets = 4e4;

for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        filename_MIP = sprintf('./4node_hs_prueba_v0/2h_budget=%d_lam=%d',bud,lam);
        load(filename_MIP);
        f_MIP = f;
        a_MIP = a;
        filename = sprintf('./4node_hs_prueba_v0_blo/bud=%d_lam=%d_alfa=%d_mu_al=%d_mu_bet=%d',bud,lam,alfa,mu_alfa,mu_beta);
        load(filename);
        disp(bud);
        fprintf('mipgap = %d',100*mipgap);
        obj_MIP = sum(sum( f_MIP.*demand.*prices )) - sum(sum( op_link_cost.*a_MIP ));
        obj = sum(sum( f.*demand.*prices )) - sum(sum( op_link_cost.*a ));

        gap = 100*(obj_MIP - obj)/obj_MIP;
       
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
end
end

function [obj_val,pax_obj,op_obj] = get_obj_val(op_link_cost,...
    prices,a,f,demand)
n = size(a,1);
pax_obj = 0;
op_obj = 0;


op_obj = sum(sum(op_link_cost.*a)); %operational costs
pax_obj = sum(sum(prices.*demand.*f));
obj_val = - pax_obj + op_obj;
end


function [s_ll,sh_ll,a_ll,f_ll,fext_ll,fij_ll,comp_time,used_budget,pax_obj,op_obj,obj_val_ll,alfa_od,beta_od,obj_hist] = compute_sim_cvx_blo(lam,alfa,n,budget,mu_alfa,mu_beta,sh_prev_in)

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidates,omega_t,omega_p] = parameters_4node_network();


debug = struct('a',{},'f',{},'fext',{},'fij',{}, ...
               'grad_alfa_v',{},'grad_beta_v',{}, ...
               'grad_alfa_f',{},'grad_beta_f',{}, ...
               'alfa_od',{},'beta_od',{});

niters = 20; %dejarlo despues asi
niters = 40;

alfa_od = ones(n);
beta_od = ones(n);

write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
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
sh_prev = sh_prev_in;
%sh_prev = [5,2,5,5,5,5];

comp_time = 0;

obj_val = 0;
obj_val_prev = 1e3;
iter = 1;
while iter <= niters
    % alfa_od = ones(n);
    % beta_od = ones(n);
    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
    write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
    stop = 0;
    for bliter=1:bliters
    
        if (abs((obj_val-obj_val_prev)./(obj_val+1e-4)) <= 1e-3) && (bliter > 1) 
            stop = 1;
        elseif (stop == 0)
        
    
    
            write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
            write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
            write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);
            
            
            %cmdpath = "cd C:\GAMS\50";
            cmdpath = "cd C:\GAMS\50";
            
            system(cmdpath);
            gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\cvx-ll.gms';
            
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
    
            
            a_ll = a;
            f_ll = f;
            fext_ll = fext;
            fij_ll = fij;

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
                station_cost,station_capacity_slope,hub_cost,link_cost,lam);

            
            
            
            
            % a_prev = 1e3*ones(n);
            % s_prev = 1e4*ones(1,n);
            % 
            % sh_prev = 1e-2*s_prev;
            % 
            % write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
            % write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
            % write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);
    
            set_max_f(n,fij,n_airlines,travel_time,prices,alt_utility,omega_p,omega_t);
               
            
            gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\cvx-sl.gms';
            
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
            
            % disp(obj_val_ll);
            
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
            alfa_od(1:n+1:end) = 1;
            beta_od(1:n+1:end) = 1;
        
            write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od);
            write_gams_param_ii('./export_txt/beta_od.txt', beta_od);
            
            % disp('beta');
            % disp(beta_od);
            % disp('alfa');
            % disp(alfa_od);
            % disp('obj_val');
            % disp(obj_val_ll);
        
            obj_hist(bliter) = obj_val_ll;
        
            set(h, 'XData', 1:bliter, ...
                   'YData', obj_hist(1:bliter));
            
            drawnow;
            obj_val_prev = obj_val;
            obj_val = obj_val_ll;

            debug(iter,bliter).a = a_ll;
            debug(iter,bliter).f = f_ll;
            debug(iter,bliter).fext = fext_ll;
            debug(iter,bliter).fij = fij_ll;
            debug(iter,bliter).grad_alfa_v = grad_alfa_v;
            debug(iter,bliter).grad_beta_v = grad_beta_v;
            debug(iter,bliter).grad_alfa_f = grad_alfa_f;
            debug(iter,bliter).grad_beta_f = grad_beta_f;
            debug(iter,bliter).alfa_od = alfa_od;
            debug(iter,bliter).beta_od = beta_od;
        end


    
    end
    if ((used_budget - budget)/budget) < 0.05
        
        if iter < (niters -1)
          %  iter = niters-1; 
            % disp('cumplo presupuesto');
        end
       % break; %comentar
    end
    s_prev = s_ll;
    sh_prev = sh_ll;
    a_prev = a_ll;
    stop = 0;
    
    iter = iter+1;
end


save('debug_matlab.mat','debug')

end




function [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,budget)

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidates,omega_t,omega_p] = parameters_4node_network();




fid = fopen("./export_txt/lam.txt",'w');
if fid < 0, error('No puedo abrir %s', fid); end
fprintf(fid, '%d', lam);
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






cmdpath = "cd C:\GAMS\50";
%cmdpath = "cd /Library/Frameworks/GAMS.framework/Resources"

system(cmdpath);

gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\mip.gms';

gamsExe = 'C:\GAMS\50\gams.exe'; 



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
    prices,a,f,demand);
req_bud = budget;
budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam);
filename = sprintf('./4node_hs_prueba_v0/2h_budget=%d_lam=%d.mat',req_bud,lam);
save(filename,'s','sprim','deltas', ...
    'a','f','fext','fij','comp_time','budget', ...
    'pax_obj','op_obj','obj_val','mipgap');

end



function set_max_f(n,fij,n_airlines,travel_time,prices,alt_utility,omega_p,omega_t)

    cotas = zeros(n);    
    for oo=1:n
        for dd=1:n 
            utility = sum(sum(travel_time(fij(:,:,oo,dd) > 1e-3)))*omega_t + prices(oo,dd)*omega_p; 
            cotas(oo,dd) = exp(utility)./( exp(utility) + n_airlines*exp(alt_utility(oo,dd)) );
        end
    end
    write_gams_param_ii('./export_txt/f_bounds.txt', cotas);
    
end

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
    a_max,candidates,omega_t,omega_p] = parameters_4node_network()

n = 4;
n_airlines = 5;

% Candidates: Full connectivity assumption
candidates = ones(n) - eye(n);

% Distances (km)
distance = readmatrix('distance.csv');

% Prices (Euros)
prices = readmatrix('prices.csv');

omega_t = -0.02;
omega_p = -0.02;



% Heuristics for other parameters
% Link Cost: proportional to distance
link_cost = 10.* distance;
link_cost(logical(eye(n))) = 1e4; % High cost for self-loops
link_cost(link_cost == 0) = 1e4; % High cost for missing connections (if any)

station_cost = 3e3.*ones(1,n); %oficinas, tasas del aeropuerto 
hub_cost = 5e3.*ones(1,n);

link_capacity_slope = 0.2 .* link_cost;
station_capacity_slope = (5*5e2 + 4*50*8).*ones(1,n); %500 e dia por estacionamiento aeronaves + 50e/hora personal*4pax
% Demand: Placeholder 10000
demand = readmatrix('demand.csv');

load_factor = 0.25 .* ones(1, n);

congestion_coef_stations = 0.1 .* ones(1, n);
congestion_coef_links = 0.1 .* ones(n);
takeoff_time = 20;
landing_time = 20;
taxi_time = 10;
cruise_time = 60 .* distance ./ 800;


travel_time = cruise_time + takeoff_time + landing_time + taxi_time; % Approx 800 km/h speed -> minutes
travel_time(logical(eye(n))) = 0;


rng(123);
p_escala = 0.4;

alt_utility  = zeros(n);


for i = 1:n
    for j = i+1:n   % SOLO mitad superior
        
        escala = rand(1, n_airlines) < p_escala;

        % --- TIEMPO ---
        alt_time_vec = travel_time(i,j) .* (1 + 0.5 .* escala) ...
                       + 60 .* escala;


        % --- PRECIO ---
        alt_price_vec = prices(i,j) ...
            + 0.3 .* prices(i,j) .* (rand(1, n_airlines) - 0.5);

        alt_u = log(sum(exp(omega_p.*alt_price_vec+omega_t*alt_time_vec))) - log(n_airlines);

        % Asignación simétrica
        alt_utility(i,j)  = alt_u;
        alt_utility(j,i)  = alt_u;
    end
end
alt_utility(1:n+1:end)  = 0;



op_link_cost = 7600 .* travel_time./60; % Proportional to distance


a_nom = 171;
tau = 0.85;
eta = 0.3;
a_max = 1e9;

end

