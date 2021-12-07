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
R, rA, rB, C1, C2, mR1, mR2, K, K1, K2, m, a, b, mu, ksi, ksm, ki0, ki1, km0, km1, k, sR, a1, a2, b1, b2, A = symbols('R rA rB C1 C2 mR1 mR2 K K1 K2 m a b mu ksi ksm ki0 ki1 km0 km1 k s_R a1 a2 b1 b2 A', positive=True, real=True)
c1A, c1B, c2, rev, koff, kR, sR0, mu, g = symbols('c1A c1B c2 rev koff kR sR0 mu g', positive=True, real=True)
gA, gB, srA, srB, a1A, a1B, b1A, b1B, b2A, b2B, K1A, K1B, K2A, K2B = symbols('gA gB srA srB a1A a1B b1A b1B b2A b2B K1A K1B K2A K2B', positive=True, real=True)

# Samples of parameter values
n = int(2E2) # Production run 1E5
ss = sobol_seq.i4_sobol_generate(11, int(n))
l = np.power(2, -3 + (4+3)*ss[:,:7])

# Two miRNAs with equal rate constants 
a1Asp, a1Bsp, a2sp, b1Asp, b1Bsp, b2Asp, b2Bsp = l[:,0], l[:,0], l[:,2], l[:,3], l[:,3], l[:,5], l[:,5]
KAsp = 10**(ss[:,-3]*(np.log10(70000)-np.log10(7)) + np.log10(7))
KBsp = 10**(ss[:,-3]*(np.log10(70000)-np.log10(7)) + np.log10(7))
gAsp = 10**((ss[:,-1]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02)))
gBsp = 10**((ss[:,-1]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02)))

# Two miRNAs with uncorrelated rate constants
#a1Asp, a1Bsp, a2sp, b1Asp, b1Bsp, b2Asp, b2Bsp = l[:,0], l[:,1], l[:,2], l[:,3], l[:,4], l[:,5], l[:,6]
#KAsp = 10**(ss[:,-4]*(np.log10(70000)-np.log10(7)) + np.log10(7))
#KBsp = 10**(ss[:,-3]*(np.log10(70000)-np.log10(7)) + np.log10(7))
#gAsp = 10**((ss[:,-2]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02)))
#gBsp = 10**((ss[:,-1]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02)))


# Define model
model_mmi2_full = {
    'pars':{
        'sR'    : 0.0,
        'b1A'   : 1.0,
        'b1B'   : 1.0,
        'b2A'   : 5,
        'b2B'   : 5,
        'a1A'   : 1.0,
        'a1B'   : 1.0,
        'a2'    : 8,
        'kR'    : 1.0,
        'gA'    : 0.2,
        'gB'    : 0.2,
        'srA'   : 0.2,
        'srB'   : 0.2,
        'K1A'   : 7000,
        'K1B'   : 7000,
        'K2A'   : 7000,
        'K2B'   : 7000,
        'koff'  : 100.0,
        'rev'   : 1,
        'w'     : 0,
        'sR0'   : 0,
    },
    'vars':{
        'rA' : \
            'srA - gA * rA + rev*koff * c1A - K1A * koff * R * rA  \
            + rev*koff * c2 - K2A * koff * c1B * rA \
            + c1A * kR * a1A + c2 * kR * a2',
        'rB' : \
            'srB - gB * rB + rev*koff * c1B - K1B * koff * R * rB  \
            + rev*koff * c2 - K2B * koff * c1A * rB \
            + c1B * kR * a1B + c2 * kR * a2',
        'R' : \
            'sR + sR0 - kR * R + rev*koff * (c1A+c1B) - koff * R * (rA*K1A + rB*K1B)  \
            + c1A * gA * b1A + c1B * gB * b1B',
        'c1A':\
            'K1A * koff * R * rA - rev*koff * c1A \
            + rev*koff * c2 - K2B * koff * c1A * rB \
            + c2 * 1 * gB * b2B - c1A * kR * a1A - c1A * gA * b1A',
        'c1B':\
            'K1B * koff * R * rB - rev*koff * c1B \
            + rev*koff * c2 - K2A * koff * c1B * rA \
            + c2 * 1 * gA * b2A - c1B * kR * a1B - c1B * gB * b1B',
        'c2':\
            'koff * (c1A*rB*K2B + c1B*rA*K2A) - rev*koff * 2*c2 \
            - c2 * (gA * b2A + gB * b2B) - c2 * kR * a2',
    },

'fns': {}, 'aux': [], 'name':'mmi2_full'}
ics_1_full = {'rA': 0.8, 'rB': 0.8, 'R': 0.1, 'c1A': 0.0, 'c1B': 0.0, 'c2': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi2_full['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['rA'], eqnD['rB'], eqnD['c1A'], eqnD['c1B'], eqnD['c2']]).jacobian(Matrix([R, rA, rB, c1A, c1B, c2]))
fJnD = lambdify((K1A, K1B, K2A, K2B, R, rA, rB, c1A, c1B, a1A, a1B, a2, b1A, b1B, b2A, b2B, kR, gA, gB, koff, rev), JnD, 'numpy')


# Tellurium object
r = model2te(model_mmi2_full, ics=ics_1_full)

uplim = 120
if 1: # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(int(n)):
        print(i)
        r['a1A'], r['a1B'], r['a2'], r['b1A'], r['b1B'], r['b2A'], r['b2B'] =  a1Asp[i], a1Bsp[i], a2sp[i], b1Asp[i], b1Bsp[i], b2Asp[i], b2Bsp[i]
        r['gA'], r['gB'], r['K1A'], r['K1B'], r['K2A'], r['K2B'] = gAsp[i], gBsp[i], KAsp[i], KBsp[i], KAsp[i], KBsp[i]
        data, bounds, boundsh = run_bf(r, auto, dirc="+", par="sR", lims=[0,uplim],
            ds=1E-2, dsmin=1E-5, dsmax=0.1)
        if data.rA.iloc[-1] < -1 or data.rB.iloc[-1] < -1:
            data, bounds, boundsh = run_bf(r, auto, dirc="+", par="sR", lims=[0,uplim],
                ds=1E-2, dsmin=1E-5, dsmax=0.01)

        data_all.append(data)

        if len(boundsh) > 0:
            print('HB point found')
            hb_cts += 1
            hbi.append(i)
            hbnds.append(boundsh)

    if 1: # Save the output
        fn = './te_data/bf_data_MMI2_DMI.tebf'

        specs = {'model':model_mmi2_full, 'n':n, 'uplim':uplim, 'KAsp':KAsp, 'KBsp':KBsp,
                'gAsp':gAsp, 'gBsp':gBsp,
                'a1Asp':a1Asp, 'a1Bsp':a1Bsp, 'a2sp':a2sp, 'b1Asp':b1Asp, 'b1Bsp':b1Bsp, 'b2Asp':b2Asp, 'b2Bsp':b2Bsp}

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)
    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    # Read a single file
    fn = './te_data/bf_data_MMI2_DMI.tebf'
    print('Reading', fn)
    with open(fn, 'rb') as f:
        f_cont = pickle.load(f)
        data_all, specs = f_cont['data_all'], f_cont['specs']
        n, uplim, KAsp, KBsp, gAsp, gBsp = specs['n'], specs['uplim'], specs['KAsp'], specs['KBsp'], specs['gAsp'], specs['gBsp']
        a1Asp, a1Bsp, a2sp, b1Asp, b1Bsp, b2Asp, b2Bsp  = specs['a1Asp'], specs['a1Bsp'], specs['a2sp'], specs['b1Asp'], specs['b1Bsp'], specs['b2Asp'], specs['b2Bsp']
    print('Curves: '+str(n)+'\t','uplim: '+str(uplim))
    for sp in ['KAsp', 'KBsp', 'gAsp', 'gBsp', 'a1Asp', 'a1Bsp', 'a2sp', 'b1Asp', 'b1Bsp', 'b2Asp', 'b2Bsp']:
        print(sp + ' is between %.4f and %.4f'%(specs[sp].min(), specs[sp].max()))
    print('\n')


# More detailed analysis of the continuation output
oui = [] # Spiral sinks
sni2 = [] # SN
hbi2 = [] # Hopf
sRhi = []
inuerr = []
for i, data in enumerate(data_all):

    if (i % 10000) == 0:
        print(i)

    if len(data) == 0:
        inuerr.append(i)
        continue

    if (data.TY == 3).sum()>0:
        sRhi.append(data.PAR[np.where(data.TY==3)[0]].mean())
        hbi2.append(i)

    if (data.TY == 2).sum()>0:
        sni2.append(i)

    Rsp, rAsp, rBsp, c1Asp, c1Bsp, c2sp = data.R.values, data.rA.values, data.rB.values, data.c1A.values, data.c1B.values, data.c2.values

    JnDsp = fJnD(KAsp[i], KBsp[i], KAsp[i], KBsp[i], Rsp, rAsp, rBsp, c1Asp, c1Bsp,
            a1Asp[i], a1Bsp[i], a2sp[i], b1Asp[i], b1Bsp[i], b2Asp[i], b2Bsp[i],
            1.0, gAsp[i], gBsp[i], 100.0, 1.0)
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
a1Astr, a1Bstr, b1Astr, b1Bstr, a2str, b2Astr, b2Bstr, gAstr, gBstr, KAstr, KBstr = r'$\it{\alpha}_1$', r"$\it{\alpha}_1$'", r"$\it{\beta}_1$", r"$\it{\beta}_1$'", r'$\it{\alpha}_2$', r"$\it{\beta}_2$", r"$\it{\beta}_2$'", r'$\it{\gamma}$', r"$\it{\gamma}$'", r"$\it{1/K}$", r"$\it{1/K}$'"
cat_strs2 = [r'Stable node for all $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$',
        r'Saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$']
cat_strs1 = [r'Stable node or saddle point for all $\it{\sigma}_R$', cat_strs2[1], cat_strs2[-2]]
dfb = pd.DataFrame({a1Astr:a1Asp[bi], a1Bstr:a1Bsp[bi], b1Astr:b1Asp[bi], b1Bstr:b1Bsp[bi], a2str:a2sp[bi], b2Astr:b2Asp[bi], b2Bstr:b2Bsp[bi], gAstr:gAsp[bi], gBstr:gBsp[bi], KAstr: KAsp[bi], KBstr: KBsp[bi], 'size':12, 'Category': cat_strs1[0], 'Initial index':bi })
dfo = pd.DataFrame({a1Astr:a1Asp[oui], a1Bstr:a1Bsp[oui], b1Astr:b1Asp[oui], b1Bstr:b1Bsp[oui], a2str:a2sp[oui], b2Astr:b2Asp[oui], b2Bstr:b2Bsp[oui], gAstr:gAsp[oui], gBstr:gBsp[oui], KAstr: KAsp[oui], KBstr: KBsp[oui], 'size':12, 'Category': cat_strs1[1], 'Initial index':oui})
dflo = pd.DataFrame({a1Astr:a1Asp[hbi2], a1Bstr:a1Bsp[hbi2], b1Astr:b1Asp[hbi2], b1Bstr:b1Bsp[hbi2], a2str:a2sp[hbi2], b2Astr:b2Asp[hbi2], b2Bstr:b2Bsp[hbi2], gAstr:gAsp[hbi2], gBstr:gBsp[hbi2], KAstr: KAsp[hbi2], KBstr: KBsp[hbi2], 'size':10, 'Category': cat_strs1[2], 'Initial index': hbi2})
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

df[a2str+'/'+a1Astr] = df.iloc[:,4] / df.iloc[:,0]
df[a2str+'/'+a1Bstr] = df.iloc[:,4] / df.iloc[:,1]
df[b2Astr+'/'+b1Astr] = df.iloc[:,5] / df.iloc[:,2]
df[b2Bstr+'/'+b1Bstr] = df.iloc[:,6] / df.iloc[:,3]

df[a2str+'/('+a1Astr+'+'+a1Bstr+')'] = df.iloc[:,4] / (df.iloc[:,0]+df.iloc[:,1])
df['('+b2Astr+'+'+b2Bstr+')/('+b1Astr+'+'+b1Bstr+')'] = (df.iloc[:,5]+df.iloc[:,6]) / (df.iloc[:,2]+df.iloc[:,3])

df[a2str+'/2'+a1Astr+'+'+a2str+'/2'+a1Bstr] = df.iloc[:,4]/2*df.iloc[:,0] + df.iloc[:,4]/2*df.iloc[:,1]
df[b2Astr+'/2'+b1Astr+'+'+b2Bstr+'/2'+b1Bstr] = df.iloc[:,5]/2*df.iloc[:,2] + df.iloc[:,6]/2*df.iloc[:,3]

df[gBstr+'/'+gAstr] = df.iloc[:,8] / df.iloc[:,7]
df[KBstr+'/'+KAstr] = df.iloc[:,10] / df.iloc[:,9]

cols = [a1Astr, a1Bstr, a2str, b1Astr, b1Bstr, b2Astr, b2Bstr, gAstr, gBstr, KAstr, KBstr]

nb = 20
bins = {a1Astr: 2**(np.linspace(np.log2(a1Asp.min()), np.log2(a1Asp.max()),nb)),
        a1Bstr: 2**(np.linspace(np.log2(a1Bsp.min()), np.log2(a1Bsp.max()),nb)),
        b1Astr: 2**(np.linspace(np.log2(b1Asp.min()), np.log2(b1Asp.max()),nb)),
        b1Bstr: 2**(np.linspace(np.log2(b1Bsp.min()), np.log2(b1Bsp.max()),nb)),
        a2str: 2**(np.linspace(np.log2(a2sp.min()), np.log2(a2sp.max()),nb)),
        b2Astr: 2**(np.linspace(np.log2(b2Asp.min()), np.log2(b2Asp.max()),nb)),
        b2Bstr: 2**(np.linspace(np.log2(b2Bsp.min()), np.log2(b2Bsp.max()),nb)),
        gAstr: 10**(np.linspace(np.log10(gAsp.min()), np.log10(gAsp.max()),nb)),
        gBstr: 10**(np.linspace(np.log10(gBsp.min()), np.log10(gBsp.max()),nb)),
        KAstr: 10**(np.linspace(np.log10(KAsp.min()), np.log10(KAsp.max()),nb)),
        KBstr: 10**(np.linspace(np.log10(KBsp.min()), np.log10(KBsp.max()),nb)),
        }

colors1 = ['gray', 'gold', 'dodgerblue']
pal1 = sns.set_palette(sns.color_palette(colors1))
colors2 = ['gray', 'gold', 'indianred', 'orange', 'dodgerblue', 'darkblue']
pal2 = sns.set_palette(sns.color_palette(colors2))

npars = len(cols)
fig, axes = plt.subplots(npars-0, npars-0, figsize=(12, 16))
fig.subplots_adjust(wspace=0.09, hspace=0.09, top=0.99, right=0.93, left=0.08, bottom=0.07)
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
            g = sns.scatterplot(data=df, x=cols[j], y=cols[i], hue='Category 2', hue_order=cat_strs2, s=5, ax=ax, legend=if_legend, alpha=0.3)
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
g = sns.JointGrid(data=df, x=df.columns[-6], y=df.columns[-5], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=4.5)
g.plot_marginals(sns.histplot, log_scale=True, hue_order=cat_strs2, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
plt.show()


sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-10], y=df.columns[-8], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=3.0)
g.plot_marginals(sns.histplot, log_scale=True, hue_order=cat_strs2, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
locmaj = mticker.LogLocator(base=10,numticks=12)
g.ax_joint.xaxis.set_major_locator(locmaj)
locmin = mticker.LogLocator(base=10.0,subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
g.ax_joint.xaxis.set_minor_locator(locmin)
g.ax_joint.xaxis.set_minor_formatter(mticker.NullFormatter())
plt.show()

sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-1], y=df.columns[-2], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=3.0)
g.plot_marginals(sns.histplot, log_scale=True, hue_order=cat_strs2, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
g.ax_joint.set_ylabel(r"$\gamma$'"+r"$/\gamma$")
g.ax_joint.set_xlabel(r"$K$'"+r"$/K$")
locmaj = mticker.LogLocator(base=10,numticks=12)
g.ax_joint.xaxis.set_major_locator(locmaj)
locmin = mticker.LogLocator(base=10.0,subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
g.ax_joint.xaxis.set_minor_locator(locmin)
g.ax_joint.xaxis.set_minor_formatter(mticker.NullFormatter())
plt.show()

sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-1], y=df.columns[-2], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=3.0)
g.plot_marginals(sns.histplot, log_scale=True, hue_order=cat_strs2, element='bars', multiple='stack')
g.plot_joint(sns.scatterplot, legend=False, s=10, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
g.ax_joint.set_ylabel(r"$\gamma$'"+r"$/\gamma$")
g.ax_joint.set_xlabel(r"$K$'"+r"$/K$")
plt.show()

