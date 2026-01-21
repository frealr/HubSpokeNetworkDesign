* ---------- Conjuntos ----------
$include "C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\param_definition_cvx.gms";



Parameters dm_pax,dm_op;

dm_pax = 1.2;
dm_op = 0.008;


*---------------------*
* Variables           *
*---------------------*


Nonnegative Variable
    s(i),        s(i),    sh(i),    
    a(i,j),
    f(o,d),      fext(o,d),
    fij(i,j,o,d)
    ;

Variable bud, op, pax, obj, ft(o,d),ftext(o,d),fr(o,d),frext(o,d),fy(o,d),fyext(o,d);  



*---------------------*
* Ecuaciones          *
*---------------------*
Equation
    bud_def      "definición de bud_obj"
    op_def       "definición de op_obj"
    pax_def      "definición de pax_obj"
    obj_def      "objetivo total"

    link_sym(i,j)
    prim_s_def(i)
    prim_a_def(i,j)

    cap_link(i,j)
    cap_station_out(i)
    cap_station_in(i)

    sum_aprim_cap
    link_flow_to_f(o,d)

    flow_start(o,d)     "conservación en origen o para cada destino d != o"
    flow_end(o,d)       "conservación en destino d para cada origen o != d"
    flow_mid(i,o,d)     "conservación en nodos intermedios i != o,d"

    zero_diag_od(o)
    zero_fij_self(i,o,d)
    zero_fij_origin(o,i,d)
    zero_fij_dest(d,i,o)
    zero_diag_odext

    split_choice(o,d)
    candidates_zero(i,j)

    fix_s_at_zero(i)
    fix_a_at_zero(i,j)
    
    expconef(o,d)
    exponefext(o,d)
    epigraphf(o,d)
    epigraphfext(o,d)
    
    def_deltasaux(i)
    def_deltaaaux(i,j)
    powercones(i)
    powerconea(i,j)
    
*rescap
;

*---------------------*
* Definición objetivo *
*---------------------*

* Presupuesto (bud): costes lineales + (si iter<niters) costes fijos aproximados
bud_def..
    bud =e= 1e-2*sum(i, station_capacity_slope(i)*(s(i) + sh(i))
         + 1e-2*sum((i,j), link_capacity_slope(i,j)*aprim(i,j))
         + (1e-2*lam*sum((i,j),
                        link_cost(i,j)*aprim(i,j)/(a_prev(i,j)+epsi))
         + 1e-2*lam*sum(i     ,
                        station_cost(i)*sprim(i)/(s_prev(i)+epsi)))$(iter<niters);

* Operación (op): coste lineal de operar enlaces
def_op_obj..
    op_obj =e=  sum((i,j), op_link_cost(i,j) * a(i,j));

* Pasajeros (pax): tiempos/precios en ruta y alternativas + entropías
pax_def..
    pax =e=
      1e-2*sum((o,d),
            demand(o,d)
          * sum((i,j), (travel_time(i,j)+prices(i,j))*fij(i,j,o,d)*logit_coef))
    + 1e-2*sum((o,d), demand(o,d)
          * (alt_time(o,d)+alt_price(o,d))*fext(o,d)*logit_coef)

    + 1e-2*sum((o,d),demand(o,d)* ft(o,d))
*          * (  ( f(o,d)*log(f(o,d)) - f(o,d) ) ))

   + 1e-2*sum((o,d), demand(o,d)*ftext(o,d))
*          * (  ( fext(o,d)*log(fext(o,d)) - fext(o,d) ) ))

*    + 1e-3*sum((o,d), demand(o,d)
*          * (  ( f(o,d)*log(f(o,d)) - f(o,d) ) ))

*   + 1e-3*sum((o,d), demand(o,d)
*          * (  ( fext(o,d)*log(fext(o,d)) - fext(o,d) ) ))
*    +1e-3*sum(i,deltast(i))
*  + 1e-3*sum(i,
*         1/(congestion_coefs_stations(i)*deltas(i) + epsi))
*    +1e-3*sum((i,j),deltaat(i,j))
*    + 1e-3*sum((i,j),
*          1/(congestion_coefs_links(i,j)*deltaa(i,j) + epsi))
;

* Objetivo total
obj_def..
    obj =e= alfa*pax/dm_pax + beta*bud + (1-alfa)*op/(dm_op);
*alfa*pax/dm_pax + beta*bud + (1-alfa)*op/(dm_op);

*---------------------*
* Restricciones       *
*---------------------*

* Definiciones primadas
prim_s_def(i)..  sprim(i) =e= s(i) + deltas(i);
prim_a_def(i,j)..aprim(i,j) =e= a(i,j) + deltaa(i,j);

* Simetría de a_prim
link_sym(i,j)$(ord(i) ne ord(j))..  aprim(i,j) =e= aprim(j,i);

* Capacidad de enlace: sum_{o,d} fij(i,j,o,d)*demand(o,d) <= tau*a(i,j)*a_nom
cap_link(i,j)..
    sum((o,d), fij(i,j,o,d)*demand(o,d))*tau =l= a(i,j)*a_nom;

* Capacidad/servicio en estación i (salidas y entradas)
cap_station_out(i)..
    sigma* sum(j,aprim(i,j) ) =l= s(i);

cap_station_in(i)..
    sigma* sum(j,aprim(j,i))  =l= s(i);

* Límite total de links construidos/operativos
sum_aprim_cap..
    sum((i,j), aprim(i,j)) =l= a_max;

* Relación f con flujos saliendo del origen
link_flow_to_f(o,d)..
    sum(j, fij(o,j,o,d)) =e= f(o,d);

* Conservación en nodos de origen (para k!=o): sum_out - sum_in = 1 - fext(o,k)
flow_start(o,d)$(ord(o) ne ord(d))..
    ( sum(j, fij(o,j,o,d)) - sum(i, fij(i,o,o,d)) ) =e= 1 - fext(o,d);

* Conservación en nodos destino (para k!=d): sum_out - sum_in = -1 + fext(k,d)
flow_end(o,d)$(ord(o) ne ord(d))..
    ( sum(j, fij(d,j,o,d)) - sum(i, fij(i,d,o,d)) ) =e= -1 + fext(o,d);

* Conservación en nodos intermedios i != o,d: sum_out - sum_in = 0
flow_mid(i,o,d)$( (ord(o) ne ord(i)) and (ord(i) ne ord(d)) )..
    ( sum(j, fij(i,j,o,d)) - sum(j, fij(j,i,o,d)) ) =e= 0;

* Zeros en diagonal y bloques prohibidos (replican fij(:,o,o,:) == 0, etc.)
zero_diag_od(o)..           f(o,o)    =l= epsi;
* fext(o,o)=0:
zero_diag_odext(o)..           fext(o,o)    =l= epsi;

* fij(i,i,*,*) = 0
zero_fij_self(i,o,d)..      fij(i,i,o,d) =l= epsi;

* fij(:,o,o,:) = 0  -> i cualquiera, j=o, o=o, d cualquiera
zero_fij_origin(o,i,d)..    fij(i,o,o,d) =l= epsi;

* fij(d,:,:,d) = 0  -> i=d, j cualquiera, o cualquiera, d=d
zero_fij_dest(d,i,o)..      fij(d,i,o,d) =l= epsi;

* Split: para o!=d, f + fext = 1
split_choice(o,d)$(ord(o) ne ord(d))..
    f(o,d) + fext(o,d) =e= 1;

* Enlaces no candidatesidatos forzados a 0: a_prim(i,j)=0 si ~(i,j) en candidates
candidates_zero(i,j)$(not candidates(i,j))..
    aprim(i,j) =e= 0;

* Fijaciones en la iteración final si previos pequeños (replica iter==niters … <=0.1)
fix_s_at_zero(i)$( (s_prev(i) <= 0.1) and (iter = niters) )..
    sprim(i) =l= epsi;

fix_a_at_zero(i,j)$( (a_prev(i,j) <= 0.1) and (iter = niters))..
    aprim(i,j) =l= epsi;
    

expconef(o,d)..  fy(o,d) =g= f(o,d)*exp(fr(o,d)/f(o,d));

exponefext(o,d)..  fyext(o,d) =g= fext(o,d)*exp(frext(o,d)/fext(o,d));

epigraphf(o,d)..  ft(o,d) =g= -fr(o,d)-f(o,d);

epigraphfext(o,d)..  ftext(o,d) =g= -frext(o,d)-fext(o,d);

def_deltasaux(i).. deltasaux(i) =e= deltas(i) + epsi;

def_deltaaaux(i,j).. deltaaaux(i,j) =e= deltaa(i,j) + epsi;

powercones(i).. sqrt(deltasaux(i)) * sqrt(deltast(i)) =g= deltasz(i);

powerconea(i,j).. deltaaaux(i,j)**0.5 * deltaat(i,j)**0.5 =g= deltaaz(i,j);

deltasz.fx(i) = 1;

*rescap(i,j,o,d).. fij(i,j,o,d) =l= capa(i,j);

*---------------------*
* Bounds básicas      *
*---------------------*
* Ya son positivas; si quieres forzar [0,1] también para f,fext,fij ya está con .up=1
* Para a, s, etc., sin cota superior explícita (ajústalas si procede)

*---------------------*
* Solve               *
*---------------------*
Model netplan /
    bud_def      
    op_def     
    pax_def      
    obj_def      

**    link_sym
    prim_s_def
    prim_a_def

    cap_link
    cap_station_out
    cap_station_in

    sum_aprim_cap
    link_flow_to_f
    flow_start   
    flow_end  
    flow_mid    

    zero_diag_od
    zero_fij_self
    zero_fij_origin
    zero_fij_dest
    zero_diag_odext

    split_choice
    candidates_zero

    fix_s_at_zero
    fix_a_at_zero
    
    expconef
    exponefext
    epigraphf
    epigraphfext


  /;


*+$ontext
s.l(i)=0.5;
sprim.l(i)=0.5;
deltas.l(i)=0.5;
a.l(i,j)=0.5;
aprim.l(i,j)=0.5;
deltaa.l(i,j)=0.5;

f.l(o,d)=0.5;
fext.l(o,d)=0.5;
fij.l(i,j,o,d)=0.5;


f.lo(o,d)=0;
fext.lo(o,d)=0;

f.up(o,d)=1;
fext.up(o,d)=1;
*$offtext

    
fy.fx(o,d)=1;
fyext.fx(o,d)=1;

deltasz.fx(i) = 1;
deltaaz.fx(i,j) = 1;




option threads = 64;

option nlp=mosek;

Solve netplan using nlp minimizing obj;
Parameter solverTime;

solverTime = netplan.resusd;

display f.l, fext.l, fij.l, a.l, s.l;



Parameter fnew(o,i,j,d);

fnew(o,i,j,d)=fij.l(i,j,o,d);

execute_unload "resultado.gdx";


file fijx /'fij_long.csv'/; put fijx;
put 'i,j,o,d,value' /;
loop((i,j,o,d),
    put i.tl ',' j.tl ',' o.tl ',' d.tl ',' fij.l(i,j,o,d):0:15 / );
putclose fijx;

EmbeddedCode Connect:
- GAMSReader:
    symbols:
      - name: aprim
- Projection:
    name: aprim.l(i,j)
    newName: aprim_level(i,j)
- ExcelWriter:
    file: output_a.xlsx
    symbols:
      - name: aprim_level
endEmbeddedCode

EmbeddedCode Connect:
- GAMSReader:
    symbols:
      - name: sprim
- Projection:
    name: sprim.l(i)
    newName: sprim_level(i)
- ExcelWriter:
    file: output_s.xlsx
    symbols:
      - name: sprim_level
endEmbeddedCode

EmbeddedCode Connect:
- GAMSReader:
    symbols:
      - name: sprim
      - name: s
      - name: deltas
      - name: aprim
      - name: a
      - name: deltaa
      - name: f
      - name: fext
      - name: fnew
      - name: solverTime
- Projection:
    name: sprim.l(i)
    newName: sprim_level(i)
- Projection:
    name: s.l(i)
    newName: s_level(i)
- Projection:
    name: deltas.l(i)
    newName: deltas_level(i)
- Projection:
    name: aprim.l(i,j)
    newName: aprim_level(i,j)
- Projection:
    name: a.l(i,j)
    newName: a_level(i,j)
- Projection:
    name: deltaa.l(i,j)
    newName: deltaa_level(i,j)
- Projection:
    name: f.l(o,d)
    newName: f_level(o,d)
- Projection:
    name: fext.l(o,d)
    newName: fext_level(o,d)
- Projection:
    name: fnew(o,i,j,d)
    newName: fij_level(o,i,j,d)
- Projection:
    name: solverTime
    newName: solver_time
- ExcelWriter:
    file: output_all.xlsx
    symbols:
      - name: sprim_level
      - name: s_level
      - name: deltas_level
      - name: aprim_level
      - name: a_level
      - name: deltaa_level
      - name: f_level
      - name: fext_level
      - name: fij_level
      - name: solver_time
endEmbeddedCode
