close all;
clear all;
clc;

% === 1) RUTA de tu GAMS ===
gamsHome = 'C:\GAMS\50';  % <-- CAMBIA esto

[basedir,~,~] = fileparts(mfilename('fullpath'));
basedir = fullfile(basedir, 'export_csv');   % subcarpeta de exportación
if ~exist(basedir,'dir'); mkdir(basedir); end

% === Datos de la red ===

[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_4node_network();

M = 1e4; %constante big M para restriccion de relacion var binaria y var continua
nreg = 40; %numero de trozos para linealizar logit
eps = 1e-3; %epsilon
n_airlines = 5; %numero de alternativas
vals_regs = linspace(0.005,0.995,nreg-1); %region a considerar en la funcion logit (de casi 0 a casi 1)
[lin_coef,bord,b] = get_linearization(n,nreg,alt_utility,vals_regs,n_airlines); %obtenemos linealizacion

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



% === 1D vectores ===
write_gams_param1d_full('./export_txt/station_cost.txt', station_cost);
write_gams_param1d_full('./export_txt/hub_cost.txt', hub_cost);
write_gams_param1d_full('./export_txt/station_capacity_slope.txt', station_capacity_slope);
write_gams_param1d_full('./export_txt/congestion_coefs_stations.txt', congestion_coef_stations);

%% Parameter definition

alfa = 0.5;
budgets = [3e4,3.5e4,4e4,4.5e4,5e4];


lam = 4;

%% run MIP

for bb=1:length(budgets)
    bud = budgets(bb);
    [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,alfa,n,bud);
end


%check the results 

for bb=1:length(budgets)
    bud = budgets(bb);
    filename = sprintf('./4node_hs_prueba_v0/bud=%d_lam=%d',bud,lam);
    load(filename);
    disp(obj_val);
end

%% run cvx model


alfas = 0.1;

for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        [s,sh,a,f,fext,fij] = compute_sim_cvx(lam,alfa,n,bud);
    end
end

% check solutions quality 

best_obj = 1e3*ones(1,length(budgets));
best_alfa = zeros(1,length(budgets));
used_bud = best_alfa;

%

for bb=1:length(budgets)
    bud = budgets(bb);
    for al=1:length(alfas)
        alfa = alfas(al);
        filename = sprintf('./4node_hs_prueba_v0_cvx/bud=%d_lam=%d_alfa=%d',bud,lam,alfa);
        load(filename);
        obj_val
        if obj_val < best_obj(bb)
                best_obj(bb) = obj_val;
                best_alfa(bb) = alfa;
                used_bud(bb) = used_budget;
        end
    end
end






%% Functions


function budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam)
budget = 0;
for i=1:n
    if (s(i) > 5e-2) && (sh(i) < 5e-2)
        budget = budget + station_cost(i)+ ...
            station_capacity_slope(i)*(s(i));
    end
    if (sh(i) > 5e-2)
        budget = budget + station_cost(i) + lam*hub_cost(i) + ...
            station_capacity_slope(i)*(sh(i)+s(i));
    end
end
end


function [obj_val,pax_obj,op_obj] = get_obj_val(op_link_cost,...
    prices,a,f,demand)

    n = size(a,1);
    pax_obj = 0;
    op_obj = 0;
    
    
    op_obj = op_obj + (sum(sum(op_link_cost.*a))); %operational costs
    for o=1:n
        for d=1:n
            pax_obj = pax_obj - prices(o,d)*demand(o,d)*f(o,d);
        end
    end
    obj_val = pax_obj + op_obj;
end




function [s,sh,a,f,fext,fij] = compute_sim_cvx(lam,alfa,n,budget)


[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_4node_network();

tic;

niters = 30;

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

disp(budget);

a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);
sh_prev = s_prev;
write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev);


cmdpath = "cd C:\GAMS\50";

system(cmdpath);
gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\cvx.gms';

gamsExe = 'C:\GAMS\50\gams.exe'; 


cmd = sprintf('%s %s ', ...
    gamsExe, gmsFile);

comp_time = 0;

for iter=1:niters
    fid = fopen("./export_txt/current_iter.txt",'w');
    if fid < 0, error('No puedo abrir %s', fid); end
    fprintf(fid, '%d', iter);
    fclose(fid);
    [status,out] = system(cmd);
    disp(out);

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

    write_gams_param_ii('./export_txt/a_prev.txt', a);
    write_gams_param1d_full('./export_txt/s_prev.txt', s);
    write_gams_param1d_full('./export_txt/sh_prev.txt', sh);


end

 gmsFile  = 'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\flow_assignment.gms';
 [status,out] = system(cmd);

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


[obj_val,pax_obj,op_obj] = get_obj_val(op_link_cost,...
prices,a,f,demand);
used_budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam);
filename = sprintf('./4node_hs_prueba_v0_cvx/bud=%d_lam=%d_alfa=%d.mat',budget,lam,alfa);
save(filename,'s','sh', ...
    'a','f','fext','fij','comp_time','used_budget', ...
    'pax_obj','op_obj','obj_val');

end



function [s,sh,a,f,fext,fij] = compute_sim_MIP(lam,alfa,n,budget)


[n,link_cost,station_cost,hub_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_utility,a_nom,tau,eta,...
    a_max,candidasourcertes,omega_t,omega_p] = parameters_4node_network();

tic;


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

disp(budget);

a_prev = 1e4*ones(n);
s_prev = 1e4*ones(1,n);
write_gams_param_ii('./export_txt/a_prev.txt', a_prev);
write_gams_param1d_full('./export_txt/s_prev.txt', s_prev);
write_gams_param1d_full('./export_txt/sh_prev.txt', s_prev);





%cmdpath = "cd C:\GAMS\50";
cmdpath = "cd C:\GAMS\50";

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

results_file_sh = readtable('./output_all.xlsx','Sheet','sh_level');
sh = table2array(results_file_sh(1,:));


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

used_budget = get_budget(s,sh,a,n,...
    station_cost,station_capacity_slope,hub_cost,link_cost,lam);
filename = sprintf('./4node_hs_prueba_v0/bud=%d_lam=%d.mat',budget,lam);
save(filename,'s','sprim','deltas', ...
    'a','f','fext','fij','comp_time','used_budget', ...
    'pax_obj','op_obj','obj_val','mipgap','sh');

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
tau = 0.8;
eta = 0.25;
a_max = 1e9;

end

