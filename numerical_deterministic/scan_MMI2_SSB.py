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
import lhsmdu
import sobol_seq
import pickle


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
Ksp = 10**(ss[:,4]*(np.log10(70000)-np.log10(7)) + np.log10(7))
gsp = 10**(ss[:,5]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02))

# Define model
model_mmi2_full = {
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
        'w'     : 0,
        'sR0'   : 0,
    },
    'vars':{
        'r' : \
            'w + (1-w)*sr - g * r + rev*koff * (c1A+c1B) - K1 * koff * 2*R * r  \
            + rev*koff * 2*c2 - K2 * koff * (c1A+c1B) * r \
            + (c1A+c1B) * kR * a1 + c2 * kR * a2 * 2',
        'R' : \
            'w + (1-w)*sR + sR0 - kR * R + rev*koff * (c1A+c1B) - K1 * koff * 2*R * r  \
            + (c1A+c1B) * g * b1',
        'c1A':\
            'K1 * koff * R * r - rev*koff * c1A \
            + rev*koff * c2 - K2 * koff * c1A * r \
            + c2 * 1 * g * b2 - c1A * kR * a1 - c1A * g * b1',
        'c1B':\
            'K1 * koff * R * r - rev*koff * c1B \
            + rev*koff * c2 - K2 * koff * c1B * r \
            + c2 * 1 * g * b2 - c1B * kR * a1 - c1B * g * b1',
        'c2':\
            'K2 * koff * (c1A+c1B) * r - rev*koff * 2*c2 \
            - c2 * 2 * g * b2 - c2 * kR * a2',
    },

'fns': {}, 'aux': [], 'name':'mmi2_full'}
ics_1_full = {'r': 0.8, 'R': 0.1, 'c1A': 0.0, 'c1B': 0.0, 'c2': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi2_full['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['r'], eqnD['c1A'], eqnD['c1B'], eqnD['c2']]).jacobian(Matrix([R, r, c1A, c1B, c2]))
fJnD = lambdify((K1, K2, R, r, c1A, c1B, a1, a2, b1, b2, kR, g, koff, rev), JnD, 'numpy')


# Tellurium object
r = model2te(model_mmi2_full, ics=ics_1_full)

uplim = 120
if 1:
    # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(int(n)):
        print(i)
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
        fn = './te_data/bf_data_MMI2_SSB.tebf'

        specs = {'model':model_mmi2_full, 'n':n, 'uplim':uplim, 'Ksp':Ksp,
                'gsp':gsp,
                'a1sp':a1sp, 'a2sp':a2sp, 'b1sp':b1sp, 'b2sp':b2sp}

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)
    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    # Reading a single file
    fn = './te_data/bf_data_MMI2_SSB.tebf'
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
snics = [] # SNIC
inuerr = []
sRthrs = []
hist_imag = np.zeros(60)
hist_hp = np.zeros(60)
hist_us = np.zeros(60)
for i, data in enumerate(data_all[:]):

    if ((i+1) % 1000) == 0:
        print(i+1)

    if len(data) == 0:
        inuerr.append(i)
        continue

    if (data.TY == 3).sum()>0:
        hp_sRs = data.PAR[np.where(data.TY==3)[0]]
        hbi2.append(i)

    if (data.TY == 2).sum()>0:
        sn_sRs = data.PAR[np.where(data.TY==2)[0]]
        sni2.append(i)

    Rsp, rsp, c1Asp, c1Bsp, c2sp = data.R.values, data.r.values, data.c1A.values, data.c1B.values, data.c2.values

    JnDsp = fJnD(Ksp[i], Ksp[i], Rsp, rsp, c1Asp, c1Bsp, a1sp[i], a2sp[i], b1sp[i], b2sp[i],
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
        rmax, imax = np.real(w).min(axis=1), np.imag(w).max(axis=1)
        imagi = np.where(imags>0)[0]
        igt = np.where(Rsp>0.1)[0]
        imagi_stab = np.where((imags>0) & ((np.real(w)>0).sum(axis=1) == 0))[0]
        unstab =  np.where((np.real(w)<0).sum(axis=1) <= (5-2))[0]
        if len(igt) > 0:
            sRthr = data.PAR[igt[0]]
            if sRthr < 0:
                continue
            sRthrs.append(sRthr)
            hs, bins = np.histogram(data.PAR[imagi_stab], bins=np.linspace(sRthr*0.0, sRthr*3.0, hist_imag.size+1))
            hist_imag = hist_imag + ((hs>0)+0)
            hs_us, bins = np.histogram(data.PAR[unstab], bins=np.linspace(sRthr*0.0, sRthr*3.0, hist_imag.size+1))
            hist_us = hist_us + ((hs_us>0)+0)
            if (len(hbi2)>0) and (i == hbi2[-1]):# and (i != sni2[-1]):
                    hs_hp, bins = np.histogram(hp_sRs, bins=np.linspace(sRthr*0.0, sRthr*3.0, hist_imag.size+1))
                    hist_hp = hist_hp + ((hs_hp>0)+0)
    if len(sni2)>0 and len(hbi2)>0 and i == sni2[-1] and i == hbi2[-1]:
        if hp_sRs.max()-sn_sRs.max() > 0:
            snics.append(i)

fig, ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2, right=0.83, left=0.17)
ax2 = ax.twinx()
ax.set_zorder(1)
ax.patch.set_visible(False)
ax2.bar(range(hist_imag.size), hist_imag/n, color='y', zorder=-10, width=1.0, alpha=0.5)
ax.bar(range(hist_hp.size), hist_hp/n, color='dodgerblue', zorder=0, width=1.0, alpha=0.7)
ax2.set_ylabel('Frequency (spiral sink)')
ax.spines['right'].set_color('y')
ax2.spines['right'].set_color('y')
ax2.yaxis.label.set_color('y')
ax2.tick_params(axis='y', colors='y')
ax.set_ylabel('Frequency (Hopf)')
ax.spines['left'].set_color('dodgerblue')
ax2.spines['left'].set_color('dodgerblue')
ax.yaxis.label.set_color('dodgerblue')
ax.tick_params(axis='y', colors='dodgerblue')
ax.set_xlabel(r'$\it{\sigma}_R$')
ax.set_xticks([0, 20, 40, 60])
ax.set_xlim(0, 40)
ltr = r'$\hat{\sigma_R}$'
ax.set_xticklabels([0, ltr, r'2$\times$'+ltr, r'3$\times$'+ltr])
plt.show()


bi = list(set(range(n)) - set(oui))
a1str, b1str, a2str, b2str, gstr, Kstr = r'$\it{\alpha}_1$', r'$\it{\beta}_1$', r'$\it{\alpha}_2$', r'$\it{\beta}_2$', r'$\it{\gamma}$', r'$\it{1/K}$'
cat_strs2 = [r'Stable node for all $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$', 
        r'Saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Spiral sink for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$',
        r'Hopf bifurcation for some $\it{\sigma}_R$'+'\n'+r'and saddle-node bifurcation for some $\it{\sigma}_R$']
cat_strs1 = [r'Stable node or saddle point for all $\it{\sigma}_R$', cat_strs2[1], cat_strs2[-2]]
dfb = pd.DataFrame({a1str:a1sp[bi], b1str:b1sp[bi], a2str:a2sp[bi], b2str:b2sp[bi], gstr:gsp[bi], Kstr: Ksp[bi], 'size':12, 'Category': cat_strs1[0], 'Initial index':bi })
dfo = pd.DataFrame({a1str:a1sp[oui], b1str:b1sp[oui], a2str:a2sp[oui], b2str:b2sp[oui], gstr:gsp[oui], Kstr: Ksp[oui], 'size':12, 'Category': cat_strs1[1], 'Initial index':oui})
dflo = pd.DataFrame({a1str:a1sp[hbi2], b1str:b1sp[hbi2], a2str:a2sp[hbi2], b2str:b2sp[hbi2], gstr:gsp[hbi2], Kstr: Ksp[hbi2], 'size':10, 'Category': cat_strs1[2], 'Initial index': hbi2})
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
        Kstr: 10**(np.linspace(np.log10(Ksp.min()), np.log10(Ksp.max()),nb)),
        }

colors1 = ['gray', 'gold', 'dodgerblue']
pal1 = sns.set_palette(sns.color_palette(colors1))
colors2 = ['gray', 'gold', 'indianred', 'orange', 'dodgerblue', 'darkblue']

npars = len(cols)
fig, axes = plt.subplots(npars, npars, figsize=(10, 9))
fig.subplots_adjust(wspace=0.09, hspace=0.09, top=0.99, right=0.93, left=0.08)
ma = np.logspace(-4, 6, 11, base=10)
mis = np.linspace(1E-4, 1E-3, 10)
mi = np.concatenate((mis, mis*1E1, mis*1E2, mis*1E3, mis*1E4, mis*1E5, mis*1E6, mis*1E7, mis*1E8, mis*1E9))
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
            g = sns.scatterplot(data=df, x=cols[j], y=cols[i], hue='Category 2', hue_order=cat_strs2, s=5, ax=ax, legend=if_legend, alpha=0.5)
            if j==0 and i==1:
                ax.legend(bbox_to_anchor=(npars + 0.5, 1.5), borderaxespad=0)
            g.set_xticks([x for x in ma])
            g.set_xticks([x for x in mi], minor = True)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            g.set_yticks([int(x) for x in ma])
            g.set_yticks([int(x) for x in mi], minor = True)
            g.set_xscale('log')
            g.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
            g.yaxis.set_major_formatter(sf)
            g.set_yscale('log')
            yl, yh = bins[cols[i]][0], bins[cols[i]][-1]
            g.set_ylim(yl, yh)
            g.set_xticks([x for x in ma])
            g.set_xticks([x for x in mi], minor = True)
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
g = sns.JointGrid(data=df, x=df.columns[-2], y=df.columns[-1], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=4.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
g.ax_joint.scatter(data=df.loc[df['Category 2'].str.contains('Hopf') & ~df['Category 2'].str.contains('saddle')], x=df.columns[-2], y=df.columns[-1], color='dodgerblue', s=2, label='Hopf only')
g.ax_joint.scatter(data=df.loc[(~df['Initial index'].isin(snics)) & df['Category 2'].str.contains('Hopf') & df['Category 2'].str.contains('saddle')], x=df.columns[-2], y=df.columns[-1], color='pink', s=2, label='SNIC')
g.ax_joint.scatter(data=df.loc[df['Initial index'].isin(snics)], x=df.columns[-2], y=df.columns[-1], color='indianred', s=2, label='Saddle-loop')
g.ax_joint.legend()
plt.show()


sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-2], y=df.columns[-1], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=4.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
sc = g.ax_joint.scatter(data=df.loc[df['Category 2'].str.contains('Hopf')], x=df.columns[-2], y=df.columns[-1], c=df.columns[4], s=2, cmap='cool', vmin=0.0, vmax=1.5)
cax = g.ax_joint.inset_axes([0.1, 0.5, 0.05, 0.4], transform=g.ax_joint.transAxes)
cbar = g.fig.colorbar(sc, ax=g.ax_joint, cax=cax)
cbar.set_ticks([0, 0.25 , 0.5, 1, 1.5])
cbar.set_ticklabels(['0', r'0.25 ($\tilde{\it{\gamma}}$)' , '0.5', '1', r'$\geq$1.5'])
cax.set_ylabel(r'$\it{\gamma}$')
plt.show()


sns.set_palette(sns.color_palette(colors2))
g = sns.JointGrid(data=df, x=df.columns[-2], y=df.columns[-1], hue='Category 2', hue_order=cat_strs2, xlim=[0.01,100],ylim=[0.01,100], height=4.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
sc = g.ax_joint.scatter(data=df.loc[df['Category 2'].str.contains('Hopf')], x=df.columns[-2], y=df.columns[-1], c=df.columns[5], s=2, cmap='cool', vmin=0, vmax=1400)
cax = g.ax_joint.inset_axes([0.1, 0.5, 0.05, 0.4], transform=g.ax_joint.transAxes)
cbar = g.fig.colorbar(sc, ax=g.ax_joint, cax=cax)
cbar.set_ticks([0, 700, 1400])
cbar.set_ticklabels(['0', r'700 (1/$\tilde{\it{K}}$)' ,r'$\geq$1.4$\times10^3$'])
cax.set_ylabel(r'$\it{1/K}$')
plt.show()

