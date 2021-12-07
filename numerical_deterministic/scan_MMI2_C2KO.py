import numpy
import matplotlib.pyplot as plt
import tellurium as te
from rrplugins import Plugin
auto = Plugin("tel_auto2000")
from te_bifurcation import model2te, run_bf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
sf = ScalarFormatter()
sf.set_scientific(False)
import re
import seaborn as sns
import os
from pickle import dump, load
from sympy import *
import sobol_seq
import pickle


# Define symbolic variables for symbolic Jacobian
R, r, C1, C2, mR1, mR2, K, K1, K2, m, a, b, mu, ksi, ksm, ki0, ki1, km0, km1, k, sR, a1, a2, b1, b2, A = symbols('R r C1 C2 mR1 mR2 K K1 K2 m a b mu ksi ksm ki0 ki1 km0 km1 k s_R a1 a2 b1 b2 A', positive=True, real=True)
c1A, c1B, c2, rev, koff, kR, sR0, mu, g = symbols('c1A c1B c2 rev koff kR sR0 mu g', positive=True, real=True)
gA, gB, srA, srB, a1A, a1B, b1A, b1B, b2A, b2B, K1A, K1B, K2A, K2B = symbols('gA gB srA srB a1A a1B b1A b1B b2A b2B K1A K1B K2A K2B', positive=True, real=True)

# Samples of parameter values
n = int(1E2) # Production run 1E5
ss = sobol_seq.i4_sobol_generate(7, int(n))
l = np.power(2, -3 + (4+3)*ss[:,:7])
a1Asp, a1Bsp, b1Asp, b1Bsp = l[:,0], l[:,1], l[:,2], l[:,3]
KAsp = 10**(ss[:,-3]*(np.log10(70000)-np.log10(7)) + np.log10(7))
KBsp = 10**(ss[:,-2]*(np.log10(70000)-np.log10(7)) + np.log10(7))
gsp = 10**((ss[:,-1]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02))-0.0)

# Define Model
model_mmi2_C2KO = {
    'pars':{
        'sR'    : 0.0,
        'b1A'   : 1.0,
        'b1B'   : 1.0,
        'b2A'   : 5,
        'b2B'   : 5,
        'a1A'   : 1.0,
        'a1B'   : 1.0,
        'kR'    : 1.0,
        'g'    : 0.2,
        'sr'    : 0.2,
        'K1A'   : 7000,
        'K1B'   : 7000,
        'koff'  : 100.0,
        'rev'   : 1,
        'w'     : 0,
        'sR0'   : 0,
    },
    'vars':{
        'r' : \
            'sr - g * r + rev*koff * (c1A+c1B) - (K1A+K1B) * koff * R * r  \
            + (c1A*a1A+c1B*a1B) * kR',
        'R' : \
            'sR + sR0 - kR * R + rev*koff * (c1A+c1B) - koff * R * (r*K1A + r*K1B)  \
            + c1B * g * b1A + c1A * g * b1B',
        'c1A':\
            'K1A * koff * R * r - rev*koff * c1A \
            - c1A * kR * a1A - c1A * g * b1A',
        'c1B':\
            'K1B * koff * R * r - rev*koff * c1B \
            - c1B * kR * a1B - c1B * g * b1B',
    },

'fns': {}, 'aux': [], 'name':'mmi2_C2KO'}

ics_1_C2KO = {'rA': 0.8, 'rB': 0.8, 'R': 0.1, 'c1A': 0.0, 'c1B': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi2_C2KO['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['r'], eqnD['c1A'], eqnD['c1B']]).jacobian(Matrix([R, r, c1A, c1B]))
fJnD = lambdify((K1A, K1B, R, r, c1A, c1B, a1A, a1B, b1A, b1B, kR, g, koff, rev), JnD, 'numpy')

# Tellurium object
r = model2te(model_mmi2_C2KO, ics=ics_1_C2KO)

uplim = 120
if 1: # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(int(n)):
        print(i)
        r['a1A'], r['a1B'], r['b1A'], r['b1B'] =  a1Asp[i], a1Bsp[i], b1Asp[i], b1Bsp[i]
        r['g'], r['K1A'], r['K1B'] = gsp[i], KAsp[i], KBsp[i]
        data, bounds, boundsh = run_bf(r, auto, dirc="+", par="sR", lims=[0,uplim],
            ds=1E-2, dsmin=1E-5, dsmax=0.1)
        if data.r.iloc[-1] < -1:
            data, bounds, boundsh = run_bf(r, auto, dirc="+", par="sR", lims=[0,uplim],
                ds=1E-2, dsmin=1E-5, dsmax=0.01)

        data_all.append(data)

        if len(boundsh) > 0:
            print('HB point found')
            hb_cts += 1
            hbi.append(i)
            hbnds.append(boundsh)

    if 1: # Save the output
        fn = './te_data/bf_data_C2KO.tebf'

        specs = {'model':model_mmi2_C2KO, 'n':n, 'uplim':uplim, 'KAsp':KAsp, 'KBsp':KBsp,
                'gsp':gsp,
                'a1Asp':a1Asp, 'a1Bsp':a1Bsp, 'b1Asp':b1Asp, 'b1Bsp':b1Bsp}

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)
    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    # Reading single file
    fn = ('./te_data/bf_data_C2KO.tebf')
    print('Reading', fn)
    with open(fn, 'rb') as f:
        f_cont = pickle.load(f)
        data_all, specs = f_cont['data_all'], f_cont['specs']
        n, uplim, KAsp, KBsp, gsp = specs['n'], specs['uplim'], specs['KAsp'], specs['KBsp'], specs['gsp']
        a1Asp, a1Bsp, b1Asp, b1Bsp = specs['a1Asp'], specs['a1Bsp'], specs['b1Asp'], specs['b1Bsp']
    print('Curves: '+str(n)+'\t','uplim: '+str(uplim))
    for sp in ['KAsp', 'KBsp', 'gsp', 'a1Asp', 'a1Bsp', 'b1Asp', 'b1Bsp']:
        print(sp + ' is between %.4f and %.4f'%(specs[sp].min(), specs[sp].max()))
    print('\n')

# More detailed analysis of the continuation output
oui = [] # Spiral sinks
sni2 = [] # SN
hbi2 = [] # Hopf
sRhi = []
inuerr = []
for i, data in enumerate(data_all):

    if len(data) == 0:
        inuerr.append(i)
        continue

    if (data.TY == 3).sum()>0:
        sRhi.append(data.PAR[np.where(data.TY==3)[0]].mean())
        hbi2.append(i)

    if (data.TY == 2).sum()>0:
        sni2.append(i)

    Rsp, rsp, c1Asp, c1Bsp, = data.R.values, data.r.values, data.c1A.values, data.c1B.values

    JnDsp = fJnD(KAsp[i], KBsp[i], Rsp, rsp, c1Asp, c1Bsp,
            a1Asp[i], a1Bsp[i], b1Asp[i], b1Bsp[i],
            1.0, gsp[i], 100.0, 1.0)
    Jsp = np.zeros((JnDsp.shape[0], JnDsp.shape[0], Rsp.shape[0]))
    for p in range(JnDsp.shape[0]):
        for q in range(JnDsp.shape[1]):
            Jsp[p,q,:] = JnDsp[p,q]
    Jsp = np.swapaxes(np.swapaxes(Jsp, 0, 2), 1,2)
    w, v = np.linalg.eig(Jsp)
    imags = (np.imag(w) != 0).sum(axis=1)
    if imags.sum() > 0:
        oui.append(i)

print(len(oui), len(hbi2), len(sni2))
