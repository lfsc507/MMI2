import argparse
import h5py
import os
import numpy as np
import tellurium as te

parser = argparse.ArgumentParser()
parser.add_argument('ic', choices=['lo', 'hi'], help='which extreme initial condition to choose')
parser.add_argument('output', type=str, help='output H5')
parser.add_argument('params', type=str, nargs='*', help='param1=value1 param2=value2 ...')
parser.add_argument('--cells', type=int, default=500, help='number of stochastic simulations')
args = parser.parse_args()

runner = te.loada('''
JM: -> M; aM * DA
JXM: M -> ; bM * M
JP: -> P; aP * M
JXP: P -> ; bP * P
JF: -> F; aF * P
JXF: F -> ; bF * F
JDXF: DR -> DA; bF * DR
JDF: DA + F -> DR; kF * F * DA / V
JDR: DR -> DA + F; kB * DR

V = 1
aM = 15.1745; aP = 1; aF = 1
bM = 1; bP = 1; bF = 1
kF = 200; kB = 50

M = 0; P = 0; F = 0
DA = 164.75; DR = 0
''')

for setting in args.params:
    param, value = setting.split('=')
    runner[param] = float(value)
v = runner['V']

runner['V'] = 1.0
extreme_ic = None
extreme_mrna = np.infty if args.ic == 'lo' else -np.infty
findextreme = np.argmin if args.ic == 'lo' else np.argmax
tcutoff = 400
for m in np.linspace(0, 500, 20):
    for p in np.linspace(0, 500, 20):
        runner.reset()
        runner['M'] = m
        runner['P'] = p
        result = runner.simulate(0, 200, 2001)
        extreme_index = findextreme(result['[M]'][tcutoff:])
        this_extreme = result['[M]'][extreme_index + tcutoff]
        if (args.ic == 'lo' and this_extreme < extreme_mrna) or (args.ic == 'hi' and this_extreme > extreme_mrna):
            extreme_ic = {f: result[f'[{f}]'][extreme_index + tcutoff] for f in runner.fs()}
            extreme_mrna = extreme_ic['M']
runner['V'] = v
#print(f'Using {args.ic} IC {extreme_ic}')

if os.path.isfile(args.output):
    with h5py.File(args.output, 'r') as file:
        data = np.array(file['results'])
        first = file.attrs['_last'] + 1
else:
    data = np.zeros((5, 2001, args.cells))
    first = 0

for c in range(first, args.cells):
    runner.reset()
    for f, conc in extreme_ic.items():
        runner[f] = round(conc * v)
    result = runner.gillespie(0, 200, 2001)
    for s, species in enumerate(['M', 'P', 'F', 'DA', 'DR']):
        data[s, :, c] = result[f'[{species}]'] / v
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('results', data=data)
        for param in runner.ps():
            hf.attrs[param] = runner[param]
        hf.attrs['_last'] = c
