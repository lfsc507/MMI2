import tellurium as te # Python-based modeling environment for kinetic models
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

tfinal = 100; 
n_timestep = 50000; 
ddt = tfinal/n_timestep
print(ddt)

# MMI2-SSB. Two C1s explicitly
ant_str = """
model simple_model   # NAME THE MODEL
# SPECIES
    species R, r, C1A, C1B, C2
# (RXN NAME:) RXN; RATE LAW
    M_in:          -> R;         sigma;
    M_out:       R ->  ;         R;
    rin:          -> r;         1;
    rout:       r -> ;          g*r;
    c1A_bind:    R + r -> C1A;   kon*R*r;
    c1B_bind:    R + r -> C1B;   kon*R*r;
    c1A_unb:     C1A -> R + r;    koff*C1A;
    c1B_unb:     C1B -> R + r;    koff*C1B;
    c1A_r:       C1A -> r;        a1*C1A;
    c1B_r:       C1B -> r;        a1*C1B;
    c1A_M:       C1A -> R;        b1*g*C1A;
    c1B_M:       C1B -> R;        b1*g*C1B;
    c2_bindA:    C1A + r -> C2;   kon*C1A*r;
    c2_bindB:    C1B + r -> C2;   kon*C1B*r;
    c2_unbA:     C2 -> C1A + r;   koff*C2;
    c2_unbB:     C2 -> C1B + r;   koff*C2;
    c2_r:       C2 -> r+r;        a2*C2;
    c2_MA:       C2 -> C1A;       b2*g*C2;
    c2_MB:       C2 -> C1B;       b2*g*C2;
# INITIAL CONDITION
    R = 3;
    r = 0.75;
    C1A = 0;
    C1B = 0;
    C2 = 0;
# PARAMETERS
    sigma =  3.58;
    g = 0.25;
    kon = 1e5;
    koff = 100;
    a1 = 1.;
    b1 = 1.;
    a2 = 12.;
    b2 = 7.;
end
"""

# MMI2-ASB
ant_str2 = """ 
model simple_model   # NAME THE MODEL
# SPECIES 
    species R, r, C1, C2
# (RXN NAME:) RXN; RATE LAW
    M_in:          -> R;         sigma; 
    M_out:       R ->  ;         R; 
    rin:          -> r;         1;
    rout:       r -> ;          g*r; 
    c1_bind:    R + r -> C1;    kon*R*r; 
    c1_unb:     C1 -> R + r;    koff*C1; 
    c1_r:       C1 -> r;        a1*C1; 
    c1_M:       C1 -> R;        b1*g*C1; 
    c2_bind:    C1 + r -> C2;   kon*C1*r;
    c2_unb:     C2 -> C1 + r;   koff*C2; 
    c2_r:       C2 -> r+r;      a2*C2; 
    c2_c1:      C2 -> C1;       b2*g*C2;  
    c2_M:       C2 -> r+R;      b2*g*C2;  
# INITIAL CONDITION
    R = 3;
    r = 0.75;
    C1 = 0;
    C2 = 0;
# PARAMETERS 
    sigma =  3.58; 
    g = 0.24; 
    kon = 1e5; 
    koff = 100; 
    a1 = 1.; 
    b1 = 1.; 
    a2 = 12.; 
    b2 = 7.; 
end
"""

r2 = te.loada(ant_str)
#r2 = te.loada(ant_str2)
# r2.kon=10;
# r2.koff=0.01;
r2.reset()
# print(r2.integrator)
results = r2.simulate(0,tfinal,n_timestep)    # simulate(time_start, time_end, number_of_points)
r2.plot(title = 'sim result', xtitle = 't', ytitle = 'Conc', figsize = (6, 4))






