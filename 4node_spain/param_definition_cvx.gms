


$setglobal TXTDIR "C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\export_txt"




Set i "nodos" /i1*i4/;
Alias (i,j,o,d);
* === Auxiliares (decláralos UNA sola vez en tu código) ===
*


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

Parameter budget
/
$include "%TXTDIR%\budget.txt"
/;


Parameter s_prev(i)
/
$include "%TXTDIR%\s_prev.txt"
/;

Parameter sh_prev(i)
/
$include "%TXTDIR%\sh_prev.txt"
/;



Scalars tau, sigma, a_nom, a_max;
tau = 0.85;
sigma = 0.3;
a_nom = 171;
a_max = 1e5;

Scalar n; n = card(i);

Scalar epsi; epsi = 1e-3;

Scalar lam /
$include "%TXTDIR%\lam.txt"
/;

Scalar alfa /
$include "%TXTDIR%\alfa.txt"
/;


Scalar iter /
$include "%TXTDIR%\current_iter.txt"
/;

Scalar niters /
$include "%TXTDIR%\niters.txt"
/;

Scalar logit_coef; logit_coef = 0.02;



display travel_time,link_cost,link_capacity_slope,prices,candidates,station_cost,station_capacity_slope,a_max, iter, niters;

