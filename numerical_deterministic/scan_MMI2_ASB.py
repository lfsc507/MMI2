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
from matplotlib import ticker as mticker


# Define symbolic variables for symbolic Jacobian
R, r, C1, C2, mR1, mR2, K, K1, K2, m, a, b, sR, ksi, ksm, ki0, ki1, km0, km1, k, sR, a1, a2, b1, b2, A =    symbols('R r C1 C2 mR1 mR2 K K1 K2 m a b sR ksi ksm ki0 ki1 km0 km1 k s_R a1 a2 b1 b2 A', positive=True, real=True)
c1A, c1B, c2, rev, koff, kR, sR0, sR, g, s, C = symbols('c1A c1B c2 rev koff kR sR0 sR g s C', positive=True, real=True)
R, r, C, mR1, mR2, K, K1, K2, m, a, b, sR, ksi, ksm, ki0, ki1, km0, km1, k, kR, A, g = \
    symbols('R r C mR1 mR2 K K1 K2 m a b sR ksi ksm ki0 ki1 km0 km1 k k_R A g', positive=True, real=True)

# Samples of parameter values
n = int(1E2) # Production run 1E5
ss = sobol_seq.i4_sobol_generate(6, int(n))
l = np.power(2, -3 + (4+3)*ss[:,:4])
a1sp, a2sp, b1sp, b2sp = l[:,0], l[:,1], l[:,2], l[:,3]
K1sp = 10**(ss[:,4]*(np.log10(70000)-np.log10(7)) + np.log10(7))
K2sp = K1sp*1
gsp = 10**(ss[:,5]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02))

# Define model
model_mmi2_1C1 = {
    'pars':{
        'sR'    : 0.0,
        'b1'    : 1.0,
        'b2'    : 5,
        'a1'    : 1.0,
        'a2'    : 8,
        'kR'    : 1.0,
        'g'    : 0.2,
        'sr'    : 0.2,
        'K1'    : 7000,
        'K2'    : 7000,
        'koff'  : 100.0,
        'rev'   : 1,
        'sR0'   : 0,
    },
    'vars':{
        'r' : \
            'sr - g * r + rev*koff * (c1A) - K1 * koff * 1*R * r  \
            + rev*koff * 1*c2 - K2 * koff * (c1A) * r \
            + (c1A) * kR * a1 + c2 * kR * a2 * 2 + c2 * g * b2',
        'R' : \
            'sR + sR0 - kR * R + rev*koff * (c1A) - K1 * koff * 1*R * r  \
            + (c1A) * g * b1 + c2 * g * b2',
        'c1A':\
            'K1 * koff * R * r - rev*koff * c1A \
            + rev*koff * c2 - K2 * koff * c1A * r \
            + c2 * 1 * g * b2 - c1A * kR * a1 - c1A * g * b1',
        'c2':\
            'K2 * koff * (c1A) * r - rev*koff * 1*c2 \
            - c2 * 2 * g * b2 - c2 * kR * a2',
    },

'fns': {}, 'aux': [], 'name':'mmi2_1C1'}
ics_1_1C1 = {'r': 0.8, 'R': 0.1, 'c1A': 0.0, 'c2': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi2_1C1['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['r'], eqnD['c1A'], eqnD['c2']]).jacobian(Matrix([R, r, c1A, c2]))
fJnD = lambdify((K1, K2, R, r, c1A,  a1, a2, b1, b2, kR, g, koff, rev), JnD, 'numpy')


# Tellurium object
r = model2te(model_mmi2_1C1, ics=ics_1_1C1)

uplim = 120
if 1: # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(int(n)):
        print(i)
        for j, p in enumerate(['a1', 'a2', 'b1', 'b2']):
            r[p] = l[i,j]
        r['g'], r['K1'], r['K2'] = gsp[i], K1sp[i], K2sp[i]
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
        fn = './te_data/bf_data_MMI2_ASB.tebf'

        specs = {'model':model_mmi2_1C1, 'n':n, 'uplim':uplim, 'K1sp':K1sp, 'K2sp':K2sp,
                'gsp':gsp,
                'a1sp':a1sp, 'a2sp':a2sp, 'b1sp':b1sp, 'b2sp':b2sp}

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)

    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    fn = './te_data/bf_data_MMI2_ASB.tebf'
    print('Reading', fn)
    with open(fn, 'rb') as f:
        f_cont = pickle.load(f)
        data_all, specs = f_cont['data_all'], f_cont['specs']
        n, uplim, K1sp, K2sp, gsp = specs['n'], specs['uplim'], specs['K1sp'], specs['K2sp'], specs['gsp']
        a1sp, a2sp, b1sp, b2sp = specs['a1sp'], specs['a2sp'], specs['b1sp'], specs['b2sp']
    print('Curves: '+str(n)+'\t','uplim: '+str(uplim))
    for sp in ['K1sp', 'K2sp', 'gsp', 'a1sp', 'a2sp', 'b1sp', 'b2sp']:
        print(sp + ' is between %.4f and %.4f'%(specs[sp].min(), specs[sp].max()))
    print('\n')

# More detailed analysis of the continuation output
oui = [] # Spiral sinks
sni2 = [] # SN
hbi2 = [] # Hopf
sRhi = []
mxi = []
inuerr = []
for i, data in enumerate(data_all):

    if (i+1 % 10000) == 0:
        print(i+1)

    if len(data) == 0:
        inuerr.append(i)
        continue

    if data.PAR.iloc[-1] < (uplim-1) or data.PAR.iloc[-1] > (uplim+1):
        mxi.append(i)

    if (data.TY == 3).sum()>0:
        sRhi.append(data.PAR[np.where(data.TY==3)[0]].mean())
        hbi2.append(i)

    if (data.TY == 2).sum()>0:
        sni2.append(i)

    Rsp, rsp, c1Asp,  c2sp = data.R.values, data.r.values, data.c1A.values, data.c2.values

    JnDsp = fJnD(K1sp[i], K2sp[i], Rsp, rsp, c1Asp, a1sp[i], a2sp[i], b1sp[i], b2sp[i],
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


bi = list(set(range(n)) - set(oui))
a1str, b1str, a2str, b2str, gstr, Kstr = r'$\it{\alpha}_1$', r'$\it{\beta}_1$', r'$\it{\alpha}_2$', r'$\it{\beta}_2$', r'$\it{\gamma}$', r'$\it{1/K}$'
cat_strs2 = [r'Stable node for all $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$',
        r'Saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$']
cat_strs1 = [r'Stable node or saddle point for all $\it{\sigma}_R$', cat_strs2[1], cat_strs2[-2]]
dfb = pd.DataFrame({a1str:a1sp[bi], b1str:b1sp[bi], a2str:a2sp[bi], b2str:b2sp[bi], gstr:gsp[bi], Kstr: K1sp[bi], 'size':12, 'Category': cat_strs1[0], 'Initial index':bi })
dfo = pd.DataFrame({a1str:a1sp[oui], b1str:b1sp[oui], a2str:a2sp[oui], b2str:b2sp[oui], gstr:gsp[oui], Kstr: K1sp[oui], 'size':12, 'Category': cat_strs1[1], 'Initial index':oui})
dflo = pd.DataFrame({a1str:a1sp[hbi2], b1str:b1sp[hbi2], a2str:a2sp[hbi2], b2str:b2sp[hbi2], gstr:gsp[hbi2], Kstr: K1sp[hbi2], 'size':10, 'Category': cat_strs1[2], 'Initial index': hbi2})
df = dfb.append(dfo, ignore_index=True).append(dflo, ignore_index=True)
df['Category 2'] = cat_strs2[0]
for i in df.index:
    ii = df.loc[:,'Initial index'][i]
    if ii in sni2:
        if ii in oui:
            if ii in hbi2:
                df.loc[i, 'Category 2'] = cat_strs2[-1]
            else:
                df.loc[i, 'Category 2'] = cat_strs2[-3]
        else:
            df.loc[i, 'Category 2'] = cat_strs2[2]
    elif ii in oui:
        if ii in hbi2:
            df.loc[i, 'Category 2'] = cat_strs2[-2]
        else:
            df.loc[i, 'Category 2'] = cat_strs2[1]

df[a2str+'/'+a1str] = df.iloc[:,2] / df.iloc[:,0]
df[b2str+'/'+b1str] = df.iloc[:,3] / df.iloc[:,1]

cols = [a1str, a2str, b1str, b2str, gstr, Kstr]

nb = 20
bins = {a1str: 2**(np.linspace(np.log2(a1sp.min()), np.log2(a1sp.max()),nb)),
        b1str: 2**(np.linspace(np.log2(b1sp.min()), np.log2(b1sp.max()),nb)),
        a2str: 2**(np.linspace(np.log2(a2sp.min()), np.log2(a2sp.max()),nb)),
        b2str: 2**(np.linspace(np.log2(b2sp.min()), np.log2(b2sp.max()),nb)),
        gstr: 10**(np.linspace(np.log10(gsp.min()), np.log10(gsp.max()),nb)),
        Kstr: 10**(np.linspace(np.log10(K1sp.min()), np.log10(K1sp.max()),nb)),
        }

colors1 = ['gray', 'gold', 'dodgerblue']
pal1 = sns.set_palette(sns.color_palette(colors1))
colors2 = ['gray', 'gold', 'indianred', 'orange', 'dodgerblue', 'darkblue']
pal2 = sns.set_palette(sns.color_palette(colors2))

npars = len(cols)
fig, axes = plt.subplots(npars, npars, figsize=(10, 9))
fig.subplots_adjust(wspace=0.09, hspace=0.09, top=0.99, right=0.93, left=0.08)
ma = np.logspace(-4, 6, 11, base=10)
mis = np.linspace(1E-4, 1E-3, 10)
mi = np.concatenate((mis, mis*1E1, mis*1E2, mis*1E3, mis*1E4, mis*1E5, mis*1E6, mis*1E7, mis*1E8, mis*1E9))
locmaj = mticker.LogLocator(base=10,numticks=12)
locmin = mticker.LogLocator(base=10.0,subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        ax = axes[i,j]
        if j>=i:
            ax.clear()
        if i == j:
            ax.set_xscale('log')
            ax.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
            ax2 = ax.twinx()
            if j==npars-1 and i==npars-1:
                if_legend = True
            else:
                if_legend = False
            stat = 'density'
            #g = sns.histplot(df, x=cols[i],hue_order=cat_strs1, hue='Category', ax=ax2, bins=20, legend=False, stat=stat, element='bars', multiple='stack')
            g = sns.histplot(df, x=cols[i],hue_order=cat_strs2, hue='Category 2', ax=ax2, bins=20, legend=False, stat=stat, element='bars', multiple='stack')
            g.set_xticks([x for x in ma])
            g.set_xticks([x for x in mi], minor = True)
            g.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.set_yticks([])
        elif j < i:
            if j==0 and i==1:
                if_legend = True
            else:
                if_legend = False
            g = sns.scatterplot(data=df, x=cols[j], y=cols[i], hue='Category 2', hue_order=cat_strs2, s=5, ax=ax, legend=if_legend, alpha=0.5)
            if j==0 and i==1:
                ax.legend(bbox_to_anchor=(npars + 0.5, 1.5), borderaxespad=0)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            g.set_xscale('log')
            ax.xaxis.set_major_locator(locmaj)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            g.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
            g.set_yscale('log')
            yl, yh = bins[cols[i]][0], bins[cols[i]][-1]
            g.set_ylim(yl, yh)
            ax.yaxis.set_major_locator(locmaj)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            g.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
        if i < (npars-1):
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if j != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if j==npars-1 and i==npars-1:
            ax.set_xlabel(cols[-1])

        if j > i:
            ax.set_axis_off()
plt.show()



sns.set_palette(sns.color_palette(colors1))
g = sns.JointGrid(data=df, x=df.columns[-2], y=df.columns[-1], hue='Category', hue_order=cat_strs1, xlim=[0.01,100],ylim=[0.01,100], height=4.5)
g.plot_marginals(sns.histplot, log_scale=True, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
plt.show()


sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-2], y=df.columns[-1], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=3.0)
g.plot_marginals(sns.histplot, log_scale=True, hue_order=cat_strs2, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
g.ax_joint.xaxis.set_major_locator(locmaj)
g.ax_joint.xaxis.set_minor_locator(locmin)
g.ax_joint.xaxis.set_minor_formatter(mticker.NullFormatter())
plt.show()

