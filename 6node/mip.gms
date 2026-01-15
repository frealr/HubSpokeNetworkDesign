* ---------- Conjuntos ----------
*$include "C:\Users\freal\MATLAB\Projects\untitled\code\6node\param_definition.gms";

$include "/Users/fernandorealrojas/Desktop/HubSpokeNetworkDesign/6node/param_definition.gms"

* ---------- Variables ----------
Variables
    obj, op_obj, pax_obj, xlen(o,d), budvar
;

Positive Variables
    s(i), sh(i), a(i,i), f(i,i), fext(i,i), zij(i,i,i,i),
    delta(seg,o,d), ghat(o,d)
;

Binary Variables
    s_bin(i),
    sh_bin(i),
    a_bin(i,i)
    z(i,i)
    fij(i,i,i,i)
    alfaseg(seg,o,d)         
;


* Cotas superiores
f.up(i,i)       = 1;
fext.up(i,i)    = 1;
zij.up(i,i,o,d) = 1;

* ---------- Ecuaciones ----------
Equations
    def_bud, def_op_obj, def_pax_obj, def_obj,
    bigM_s, bigM_sh, bigM_a, relssh, bud_avail,
    link_cap, station_cap, hub_connect_cap, hub_direct_cap,
    tot_links, link_sym,
    fij_od_flow, fij_do_flow, fij_trans
    diag_zero_fo, fij_zero_diag, fij_zero_oo, fij_zero_dd
    split_fd, f_leq_Mz
    cand_zero
    zij_bigM_lb, zij_leq_f, zij_leq_Mfij
    def_xlen, alfaseg_sum, delta_bound,delta_bound_last, xlen_interp, slice_bound, ghat_interp, f_leq_ghat
;


def_bud.. budvar =e= sum(i, station_capacity_slope(i)*(s(i)+sh(i)))
         + sum((i,j),
                        link_cost(i,j)*a_bin(i,j))
         + sum(i,
                        station_cost(i)*s_bin(i))
         + lam*sum(i,
                        hub_cost(i)*sh_bin(i));

* ---------- Objetivo ----------
def_op_obj..
    op_obj =e=  sum((i,j), op_link_cost(i,j) * a(i,j));

def_pax_obj..
    pax_obj =e= - sum((o,d), prices(o,d) * demand(o,d) * f(o,d));

def_obj..
    obj =e=  pax_obj + op_obj;
    


* ---------- Restricciones de infraestructura ----------
*sprim_link(i)..          s_prim(i)   =e= s(i) + delta_s(i);
*aprim_link(i,j)..        a_prim(i,j) =e= a(i,j) + delta_a(i,j);

bigM_s(i)..              s(i)   =l= M * s_bin(i);
bigM_sh(i)..             sh(i)  =l= M * sh_bin(i);
bigM_a(i,j)..            a(i,j) =l= M * a_bin(i,j);
relssh(i)..              sh_bin(i) =l= M*s_bin(i);

bud_avail..
    budvar =l= budget;

link_cap(i,j)..
     sum((o,d), zij(i,j,o,d) * demand(o,d)) =l= a(i,j) * a_nom * tau;

station_cap(i)..   sigma * (sum(j, a(j,i)) + sum(j, a(i,j))) =l= s(i) + sh(i);

hub_connect_cap(i).. sigma * (sum( (o,d,j)$(ord(o)<>ord(i)),
    demand(o,d) * fij(i,j,o,d) / (a_nom * tau))  + sum( (o,d,j)$(ord(d)<>ord(i)),
    demand(o,d) * fij(j,i,o,d) / (a_nom * tau))) =l= sh(i);
    
hub_direct_cap(o,d).. sigma * demand(o,d) * fij(o,d,o,d) / (a_nom * tau) =l= sh(o) + sh(d);

tot_links.. sum((i,j), a(i,j)) =l= a_max;

link_sym(i,j).. a(i,j) =e= a(j,i);

* ---------- Restricciones de flujo ----------
fij_od_flow(o,d)$(ord(o)<>ord(d))..
    sum(j$(ord(j)<>ord(o)), fij(o,j,o,d))
  - sum(i$(ord(i)<>ord(o)), fij(i,o,o,d))
  =e= z(o,d);

fij_do_flow(o,d)$(ord(o)<>ord(d))..
    sum(j$(ord(j)<>ord(d)), fij(d,j,o,d))
  - sum(i$(ord(i)<>ord(d)), fij(i,d,o,d))
  =e= -z(o,d);

fij_trans(i,o,d)$(ord(i)<>ord(o) and ord(i)<>ord(d))..
    sum(j$(ord(j)<>ord(i)), fij(i,j,o,d))
  - sum(ii$(ord(ii)<>ord(i)), fij(ii,i,o,d))
  =e= 0;

diag_zero_fo(o)..            f(o,o) + fext(o,o) =e= 0;
fij_zero_diag(i,o,d)..       fij(i,i,o,d) =e= 0;
fij_zero_oo(o,i,d)..         fij(i,o,o,d) =e= 0;
fij_zero_dd(d,i,o)..         fij(d,i,o,d) =e= 0;

split_fd(o,d)$(ord(o)<>ord(d))..  f(o,d) + fext(o,d) =e= 1;

f_leq_Mz(o,d)$(ord(o)<>ord(d))..  f(o,d) =l=  z(o,d);

cand_zero(i,j)$(candidates(i,j)=0)..  a(i,j) =e= 0;

zij_bigM_lb(i,j,o,d)..  zij(i,j,o,d) =g= f(o,d) - (1 - fij(i,j,o,d));
zij_leq_f(i,j,o,d)..    zij(i,j,o,d) =l= f(o,d);
zij_leq_Mfij(i,j,o,d).. zij(i,j,o,d) =l= fij(i,j,o,d);

* ---------- PWL con convex combination + binarios ----------
def_xlen(o,d)$(ord(o)<>ord(d))..
    xlen(o,d) =e= sum((i,j), fij(i,j,o,d) * logit_coef*(-travel_time(i,j))) + f(o,d)*logit_coef*(-prices(o,d));
    
alfaseg_sum(o,d)$(ord(o)<>ord(d)).. sum(seg,alfaseg(seg,o,d)) =e= 1;

delta_bound(seg,o,d)$((ord(o)<>ord(d)) and (ord(seg) lt card(seg))).. delta(seg,o,d) =l= alfaseg(seg,o,d)*(b(seg+1,o,d)-b(seg,o,d));

delta_bound_last(seg,o,d)$(ord(seg)=card(seg))..  delta(seg,o,d) =l= alfaseg(seg,o,d)*(-b(seg,o,d));

xlen_interp(o,d)$(ord(o)<>ord(d)).. xlen(o,d) =e= sum(seg,alfaseg(seg,o,d)*b(seg,o,d) + delta(seg,o,d));

slice_bound(seg,o,d)$((ord(seg) lt card(seg)) and (ord(o)<>ord(d)) ).. b(seg,o,d) + delta(seg,o,d) =l= b(seg+1,o,d);

ghat_interp(o,d)$(ord(o)<>ord(d)).. ghat(o,d) =e= sum(seg,alfaseg(seg,o,d)*bord(seg,o,d) + mreg(seg,o,d)*delta(seg,o,d));   

f_leq_ghat(o,d)$(ord(o)<>ord(d))..     f(o,d) =l= ghat(o,d);


* ---------- Modelo y resolución ----------
Model netdesign /
    def_bud, def_op_obj, def_pax_obj, def_obj,
    bigM_s, bigM_sh, bigM_a, relssh, bud_avail,
    link_cap, station_cap, hub_connect_cap, hub_direct_cap,
    tot_links, link_sym,
    fij_od_flow, fij_do_flow, fij_trans
    diag_zero_fo, fij_zero_diag, fij_zero_oo, fij_zero_dd
    split_fd, f_leq_Mz
    cand_zero
    zij_bigM_lb, zij_leq_f, zij_leq_Mfij 
    def_xlen
    alfaseg_sum
    delta_bound
    delta_bound_last
    slice_bound
    xlen_interp
    ghat_interp, f_leq_ghat
/;


option threads = 60;
option mip     = mosek;

option reslim = 1800;

Parameter mipgap;

Solve netdesign using mip minimizing obj;

* ---------- Salidas básicas ----------
*Display f.l, fext.l, fij.l, xlen.l, ghat.l,
*        obj.l, op_obj.l, pax_obj.l,
*        s.l, s_bin.l, a.l, a_bin.l, z.l, b;

mipgap = abs(netdesign.objval - netdesign.objest) / (1e-9 + abs(netdesign.objval));

Parameter solverTime;

solverTime = netdesign.resusd;

Display b,budget,netdesign.objval, netdesign.objest, mipgap;


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
      - name: a
- Projection:
    name: a.l(i,j)
    newName: a_level(i,j)
- ExcelWriter:
    file: output_a.xlsx
    symbols:
      - name: a_level
endEmbeddedCode

EmbeddedCode Connect:
- GAMSReader:
    symbols:
      - name: s
- Projection:
    name: s.l(i)
    newName: s_level(i)
- ExcelWriter:
    file: output_s.xlsx
    symbols:
      - name: s_level
endEmbeddedCode

EmbeddedCode Connect:
- GAMSReader:
    symbols:
      - name: s
      - name: sh
      - name: a
      - name: f
      - name: fext
      - name: fnew
      - name: mipgap
      - name: solverTime
- Projection:
    name: s.l(i)
    newName: s_level(i)
- Projection:
    name: sh.l(i)
    newName: sh_level(i)
- Projection:
    name: a.l(i,j)
    newName: a_level(i,j)
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
    name: mipgap
    newName: mip_opt_gap
- Projection:
    name: solverTime
    newName: solver_time
- ExcelWriter:
    file: output_all.xlsx
    symbols:
      - name: s_level
      - name: sh_level
      - name: a_level
      - name: f_level
      - name: fext_level
      - name: fij_level
      - name: mip_opt_gap
      - name: solver_time      
endEmbeddedCode