import numpy
import matplotlib.pyplot as plt
import tellurium as te
from rrplugins import Plugin
auto = Plugin("tel_auto2000")
from te_bifurcation import model2te, run_bf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
R, r, C, mR1, mR2, K, K1, K2, m, a, b, sR, ksi, ksm, ki0, ki1, km0, km1, k, kR, A = \
    symbols('R r C mR1 mR2 K K1 K2 m a b sR ksi ksm ki0 ki1 km0 km1 k k_R A', positive=True, real=True)


# Samples of parameter values
n = int(1E2) # Production run 1E5
ss = sobol_seq.i4_sobol_generate(4, n)
l = np.power(2, -3 + (4+3)*ss[:,:2])
a1sp, b1sp = l[:,0], l[:,1]
Ksp = 10**(ss[:,-2]*(np.log10(70000)-np.log10(7)) + np.log10(7))
gsp = 10**(ss[:,-1]*(np.log10(2)-np.log10(0.02)) + np.log10(0.02))


# Model definition
model_mmi1_full = {
    'pars': {
        'sR': 0.0,
        'a1' : 1,
        'b1' : 1,
        'sR0': 0.0,
        'g': 1.0,
        'K' : 10000,
        'koff': 100,
    },
    'vars': {
        'r': '1 - koff*K*R*r + koff*C - g*r + a1*C',
        'R': 'sR0 + sR - koff*K*R*r + koff*C - R + b1*g*C',
        'C': 'koff*K*R*r - koff*C - a1*C - b1*g*C',
    },

    'fns': {}, 'aux': [], 'name': 'mmi1_full'}
ics_1_mmi1_full = {'r': 0.9, 'R': 0.0, 'C': 0.0}

# Symbolic Jacobian
eqnD = {}
for k, v in model_mmi1_full['vars'].items():
    eqnD[k] = parsing.sympy_parser.parse_expr(v, locals())
JnD = Matrix([eqnD['R'], eqnD['r'], eqnD['C']]).jacobian(Matrix([R, r, C]))
fJnD = lambdify((K, R, r, C, a1, b1, g, koff), JnD, 'numpy')

# Tellurium object
r = model2te(model_mmi1_full, ics=ics_1_mmi1_full)

uplim = 120
if 1:
    # A new run
    hb_cts, hbi, hbnds = 0, [], []
    data_all = []
    inuerr = []
    for i in range(n):
        print(i)
        for j, p in enumerate(['a1', 'b1']):
            r[p] = l[i,j]
        r['g'], r['K'] = gsp[i], Ksp[i]
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
        fn = './te_data/bf_data_MMI1.tebf'

        specs = {'model':model_mmi1_full, 'n':n, 'uplim':uplim, 'Ksp':Ksp,
                'gsp':gsp,
                'a1sp':a1sp, 'b1sp':b1sp }

        with open(fn, 'wb') as f:
            pickle.dump({'data_all': data_all, 'specs': specs}, f)
    print('Sets with HB: ', hb_cts)
    print('Numerical errors', len(inuerr))
else:
    # Reading a single file
    fn = './te_data/bf_data_MMI1.tebf'
    print('Reading', fn)
    with open(fn, 'rb') as f:
        f_cont = pickle.load(f)
        data_all, specs = f_cont['data_all'], f_cont['specs']
        n, uplim, Ksp, gsp = specs['n'], specs['uplim'], specs['Ksp'], specs['gsp']

        a1sp, b1sp = specs['a1sp'], specs['b1sp']
    print('Curves: '+str(n)+'\t','uplim: '+str(uplim))
    for sp in ['Ksp', 'gsp', 'a1sp', 'b1sp']:
        print(sp + ' is between %.4f and %.4f'%(specs[sp].min(), specs[sp].max()))
    print('\n')

# More detailed analysis of the continuation output
oui = [] # Spiral sinks
hbi = [] # Hopf
mxi = [] # Hopf and SN
inuerr = []
binned_Rs = []
binned_Rts = []
binned_cons = []
hist_imag = np.zeros(60)
nR = 62
do_pars = []
for i, data in enumerate(data_all[:]):

    if ((i+1) % 10000) == 0:
        print(i+1)

    if len(data) == 0:
        inuerr.append(i)
        continue

    if data.PAR.iloc[-1] < (uplim-1) or data.PAR.iloc[-1] > (uplim+1):
        mxi.append(i)

    if (data.TY == 3).sum()>0:
        hbi.append(i)

    Rsp, rsp, Csp = data.R.values, data.r.values, data.C.values
    JnDsp = fJnD(Ksp[i], Rsp, rsp, Csp, a1sp[i], b1sp[i], gsp[i], 100.0)
    Jsp = np.zeros((JnDsp.shape[0], JnDsp.shape[0], Rsp.shape[0]))
    for p in range(JnDsp.shape[0]):
        for q in range(JnDsp.shape[1]):
            Jsp[p,q,:] = JnDsp[p,q]
    Jsp = np.swapaxes(np.swapaxes(Jsp, 0, 2), 1,2)
    w, v = np.linalg.eig(Jsp)
    #print(w)
    if_imag = np.imag(w) != 0
    imags = ((if_imag).sum(axis=1)>0) & (Rsp>-10) & (rsp>-10)
    igt = np.where(Rsp>0.01)[0]
    if (len(igt) > 0):
        sRthr = data.PAR[igt[0]]
        std_sigs = np.linspace(sRthr*0.0, sRthr*3.1, nR)
        ids = np.searchsorted(data.PAR, std_sigs)
        binned_R, binned_Rt = np.empty(nR), np.empty(nR)
        binned_R[:], binned_Rt[:] = np.NaN, np.NaN
        R_data = Rsp[[x for x in ids if x < Rsp.size]]
        Rt_data = R_data + Csp[[x for x in ids if x < Rsp.size]]
        binned_R[:R_data.size] = R_data
        binned_Rt[:R_data.size] = Rt_data
        binned_Rs.append(binned_R)
        binned_Rts.append(binned_Rt)
        binned_cons.append(std_sigs)

    if imags.sum() > 0:
        if (a1sp[i]>1 and b1sp[i]>1) or (a1sp[i]<1 and b1sp[i]<1):
            continue
        rmax, imax = np.real(w).min(axis=1), np.imag(w).max(axis=1)
        oui.append(i)
        imagi = np.where(imags>0)[0]
        if len(igt) > 0:
            hs, bins = np.histogram(data.PAR[imagi], bins=np.linspace(sRthr*0.0, sRthr*3.0, hist_imag.size+1))
            hist_imag = hist_imag + ((hs>0)+0)


fig, ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2, right=0.78, left=0.15)
ax2 = ax.twinx()
ax2.bar(range(hist_imag.size), hist_imag/n, color='y', zorder=-10, width=1.0, alpha=0.5)
dfl = pd.DataFrame(binned_Rs).melt()
sns.lineplot(x="variable", y="value", data=dfl, color='k', ax=ax, ci=99.9, palette="flare")
ax.set_ylabel(r'Steady state $\it{R}$ (A.U.)')
ax.set_xlabel(r'$\sigma_R$')
ax.set_xticks([0, 20, 40, 60])
ltr = r'$\hat{\it{\sigma_R}}$'
ax.set_xticklabels([0, ltr, r'2$\times$'+ltr, r'3$\times$'+ltr])
ax.set_xlim(0, 40)
ax.spines['right'].set_color('y')
ax2.spines['right'].set_color('y')
ax2.yaxis.label.set_color('y')
ax2.tick_params(axis='y', colors='y')
ax2.set_ylabel(r'Frequency (spiral sink)')
plt.show()

figc, axc = plt.subplots(figsize=(4,3))
figc.subplots_adjust(bottom=0.2, right=0.90, left=0.25)
sns.lineplot(x="variable", y="value", data=dfl, color='k', ax=axc, ci=99.9, palette="flare", label=r'$\it{R}$')
dft = pd.DataFrame(binned_Rts).melt()
sns.lineplot(x="variable", y="value", data=dft, color='m', ax=axc, ci=99.9, palette="flare", label=r'$\it{R}\rm{_T}$')
dfc = pd.DataFrame(binned_cons).melt()
sns.lineplot(x="variable", y="value", data=dfc, color='b', ax=axc, ci=99.9, palette="flare", label=r'$\it{R}\rm{_T} (=\it{R})$'+'\nw/o microRNA')
axc.set_ylabel('Steady state mRNA\nconcentration (A.U.)')
axc.set_xlabel(r'$\sigma_R$')
axc.set_yscale('log')
axc.set_ylim(1E-4, 60)
axc.set_xticks([0, 20, 40, 60])
ltr = r'$\hat{\it{\sigma_R}}$'
axc.set_xticklabels([0, ltr, r'2$\times$'+ltr, r'3$\times$'+ltr])
axc.set_xlim(0, 40)
axc.set_yscale('log')
axc.set_ylim(1E-4, 60)
axc.legend()
plt.show()


bi = list(set(range(n)) - set(oui))
astr, bstr, gstr, Kstr = r'$\it{\alpha}$', r'$\it{\beta}$', r'$\it{\gamma}$', r'$1/\it{K}$'
cat_strs = [r'Stable node for all $\it{\sigma_R}$', r'Spiral sink for some $\it{\sigma_R}$']
dfb = pd.DataFrame.from_dict({astr:a1sp[bi], bstr:b1sp[bi], gstr:gsp[bi], Kstr: Ksp[bi],'Category': cat_strs[0]})
dfo = pd.DataFrame.from_dict({astr:a1sp[oui], bstr:b1sp[oui], gstr:gsp[oui], Kstr: Ksp[oui], 'Category':  cat_strs[1]})
df = dfb.append(dfo, ignore_index=True)
cols = [astr, bstr, gstr, Kstr]

nb = 20
bins = {astr: 2**(np.linspace(np.log2(a1sp.min()), np.log2(a1sp.max()),nb)),
        bstr: 2**(np.linspace(np.log2(b1sp.min()), np.log2(b1sp.max()),nb)),
        gstr: 10**(np.linspace(np.log10(gsp.min()), np.log10(gsp.max()),nb)),
        Kstr: 10**(np.linspace(np.log10(Ksp.min()), np.log10(Ksp.max()),nb)),
        }

colors = ['gray', 'gold']
sns.set_palette(sns.color_palette(colors))

npars = len(cols)
fig, axes = plt.subplots(npars, npars, figsize=(10, 9))
fig.subplots_adjust(wspace=0.09, hspace=0.09, top=0.99, right=0.93)
ma = np.logspace(-4, 6, 11, base=10)
mi = np.linspace(1E-4, 1E3, 11)
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
            g = sns.histplot(df, x=cols[i],hue_order=cat_strs, hue='Category', ax=ax2, bins=20, legend=False, stat=stat, multiple='stack')
            #sns.histplot(df[df.Category.str.contains('node')], x=cols[i], ax=ax2, bins=20, legend=False, stat='density', element='step')
            #sns.histplot(df[df.Category.str.contains('sink')], x=cols[i], ax=ax2, bins=20, legend=False, stat='density', color='gold', element='step')
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
            sns.scatterplot(data=df, x=cols[j], y=cols[i], hue='Category', s=12, ax=ax, legend=if_legend, alpha=0.5)
            if j==0 and i==1:
                ax.legend(bbox_to_anchor=(npars + 0.1, 1), borderaxespad=0)
            ax.set_yticks([int(x) for x in ma])
            ax.set_yticks([int(x) for x in mi], minor = True)
            #ax.text(0.5,0.9, f"{i}, {j}", ha="center", transform=ax.transAxes)
            ax.set_xscale('log')
            ax.set_xlim(bins[cols[j]][0], bins[cols[j]][-1])
            ax.yaxis.set_major_formatter(sf)
            ax.set_yscale('log')
            yl, yh = bins[cols[i]][0], bins[cols[i]][-1]
            ax.set_ylim(yl, yh)
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
fig.subplots_adjust(left=0.067, bottom=0.082)
plt.show()



hbi_tt = hbi
sns.set_palette(sns.color_palette(colors))
g = sns.JointGrid(data=df, x=df.columns[0], y=df.columns[1], hue='Category', hue_order=cat_strs, xlim=[0.125,16],ylim=[0.125,16], height=3.0)
g.plot_marginals(sns.histplot, log_scale=True, multiple='stack', stat=stat)
g.plot_joint(sns.scatterplot, legend=False, s=5, alpha=0.5)
g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
plt.show()


