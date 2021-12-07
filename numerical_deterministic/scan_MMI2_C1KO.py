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
R, r, C1, C2, mR1, mR2, K, K1, K2, m, a, b, sR, ksi, ksm, ki0, ki1, km0, km1, k, sR, a1, a2, b1, b2, A =    symbols('R r C1 C2 mR1 mR2 K K1 K2 m a b sR ksi ksm ki0 ki1 km0 km1 k s_R a1 a2 b1 b2 A', positive=True, real=True)
c1A, c1B, c2, rev, koff, kR, sR0, sR, g = symbols('c1A c1B c2 rev koff kR sR0 sR g', positive=True, real=True)

# Samples of parameter values
n = int(1E2) # Production run 1E5
ss = sobol_seq.i4_sobol_generate(6, int(n))
l = np.power(2, -3 + (4+3)*ss[:,:4])
a1sp, a2sp, b1sp, b2sp = l[:,0], l[:,1], l[:,2], l[:,3]
Ksp = 10**(ss[:,4]*(np.log10(70000)-np.log10(7)) + np.log10(7))
gsp = 10**(ss[:,5]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02))

# Define Model
model_mmi2_C1KO = {
    'pars':{
        'sR'    : 0.0,
        'b1'    : 1.0,
        'b2'    : 5,
        'a1'    : 1.0,
        'a2'    : 8,
        'kR'    : 1.0,
        'g'     : 0.2,
        'sr'    : 0.2,
        'K1'    : 7000,
        'K2'    : 7000,
        'koff'  : 100.0,
        'rev'   : 1,
        'sR0'   : 0,
    },
    'vars':{
        'r' : \
            'sr - g * r  \
            + rev*koff * 2*c2 - 2 * K2 * koff * R * r * r \
            + c2 * kR * a2 * 2 + c2 * g * b2 * 2',
        'R' : \
            'sR + sR0 - kR * R + rev*koff * c2 - K2 * koff * R * r * r  \
            + c2 * 2 * g * b2',
        'c2':\
            'K2 * koff * R * r * r - rev*koff * 1*c2 \
            - c2 * 2 * g * b2 - c2 * kR * a2',
    },

'fns': {}, 'aux': [], 'name':'mmi2_C1KO'}
ics_1_C1KO = {'r': 0.8, 'R': 0.1, 'c2': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi2_C1KO['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['r'], eqnD['c2']]).jacobian(Matrix([R, r, c2]))
fJnD = lambdify((K1, K2, R, r,  a1, a2, b1, b2, kR, g, koff, rev), JnD, 'numpy')


# Tellurium object
r = model2te(model_mmi2_C1KO, ics=ics_1_C1KO)

uplim = 120
if 1: # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(int(n)):
        print(i)
        if i in [860]: # Numerical error
            data_all.append([])
            continue
        for j, p in enumerate(['a1', 'a2', 'b1', 'b2']):
            r[p] = l[i,j]
        r['g'], r['K1'], r['K2'] = gsp[i], Ksp[i], Ksp[i]
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
        fn = './te_data/bf_data_C1KO.tebf'

        specs = {'model':model_mmi2_C1KO, 'n':n, 'uplim':uplim, 'Ksp':Ksp,
                'gsp':gsp,
                'a1sp':a1sp, 'a2sp':a2sp, 'b1sp':b1sp, 'b2sp':b2sp}

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)

    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    fn = './te_data/bf_data_C1KO.tebf'
    print('Reading', fn)
    with open(fn, 'rb') as f:
        f_cont = pickle.load(f)
        data_all, specs = f_cont['data_all'], f_cont['specs']
        n, uplim, Ksp, gsp = specs['n'], specs['uplim'], specs['Ksp'], specs['gsp']

        a1sp, a2sp, b1sp, b2sp = specs['a1sp'], specs['a2sp'], specs['b1sp'], specs['b2sp']
    print('Curves: '+str(n)+'\t','uplim: '+str(uplim))
    for sp in ['Ksp', 'gsp', 'a1sp', 'a2sp', 'b1sp', 'b2sp']:
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

    Rsp, rsp, c2sp = data.R.values, data.r.values, data.c2.values
    JnDsp = fJnD(Ksp[i], Ksp[i], Rsp, rsp, a1sp[i], a2sp[i], b1sp[i], b2sp[i],
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
