

$setglobal TXTDIR  "C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\export_txt"


Sets
    i   / i1*i6 /
;
Alias (i,j,o,d,ii,jj);

* Breakpoints para PWL (K puntos => K-1 segmentos)
Set seg / seg1*seg40 /;
* segmentos entre breakpoints (K=20 => 19 segmentos)


Parameter b(seg,o,d)
/
$include "%TXTDIR%\b.txt"
/;


Parameter bord(seg,o,d)
/
$include "%TXTDIR%\bord.txt"
/;

Parameter mreg(seg,o,d)
/
$include "%TXTDIR%\lin_coef.txt"
/;


Parameter demand(o,d)

/
$include "%TXTDIR%\demand.txt"
/;

Parameter alt_utility(o,d)

/
$include "%TXTDIR%\alt_utility.txt"
/;

* --- (i,j) ---
Parameter link_cost(i,j)
/
$include "%TXTDIR%\link_cost.txt"
/;

Parameter link_capacity_slope(i,j)
/
$include "%TXTDIR%\link_capacity_slope.txt"
/;

Parameter prices(o,d)
/
$include "%TXTDIR%\prices.txt"
/;

Parameter op_link_cost(i,j)
/
$include "%TXTDIR%\op_link_cost.txt"
/;

Parameter congestion_coefs_links(i,j)
/
$include "%TXTDIR%\congestion_coefs_links.txt"
/;

Parameter timelimit
/
30
/;

Parameter candidates(i,j)
/
$include "%TXTDIR%\candidates.txt"
/;

Parameter travel_time(i,j)
/
$include "%TXTDIR%\travel_time.txt"
/;

* --- (i) ---
Parameter station_cost(i)
/
$include "%TXTDIR%\station_cost.txt"
/;

Parameter hub_cost(i)
/
$include "%TXTDIR%\hub_cost.txt"
/;

Parameter station_capacity_slope(i)
/
$include "%TXTDIR%\station_capacity_slope.txt"
/;

Parameter congestion_coefs_stations(i)
/
$include "%TXTDIR%\congestion_coefs_stations.txt"
/;


Parameter a_prev(i,j)
/
$include "%TXTDIR%\a_prev.txt"
/;

Parameter s_prev(i)
/
$include "%TXTDIR%\s_prev.txt"
/;

Parameter sh_prev(i)
/
$include "%TXTDIR%\sh_prev.txt"
/;


Scalars tau, sigma, a_nom, a_max, M, n_airlines;
tau = 0.85;
sigma = 0.3;
a_nom = 171;
a_max = 1e7;
M = 1e5;
n_airlines = 5;

Scalar n; n = card(i);

Scalar epsi; epsi = 1e-3;

Scalar lam /
$include "%TXTDIR%\lam.txt"
/;

Scalar alfa /
$include "%TXTDIR%\alfa.txt"
/;


Scalar dm_pax /
$include "%TXTDIR%\dm_pax.txt"
/;

Scalar dm_op /
$include "%TXTDIR%\dm_op.txt"
/;

Scalar budget /
$include "%TXTDIR%\budget.txt"
/;

Scalar nom_bud
/
$include "%TXTDIR%\budget.txt"
/;

Scalar logit_coef; logit_coef = 0.02;



display travel_time,link_cost,link_capacity_slope,prices,candidates,station_cost,station_capacity_slope,a_max,demand;

