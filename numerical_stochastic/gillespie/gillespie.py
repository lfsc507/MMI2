import argparse
import h5py
import os
import numpy as np
import tellurium as te

parser = argparse.ArgumentParser()
parser.add_argument('ic', choices=['lo', 'hi'], help='which extreme initial condition to choose')
parser.add_argument('output', type=str, help='output H5')
parser.add_argument('params', type=str, nargs='*', help='param1=value1 param2=value2 ...')
parser.add_argument('--cells', type=int, default=200, help='number of stochastic simulations')
args = parser.parse_args()

runner = te.loada('''
JSI: -> miRNA; s_i * V
JDI: miRNA -> ; k_i * miRNA
JSM: -> mRNA; (s_m + s_m0) * V
JDM: mRNA -> ; k_m * mRNA
JC1A: mRNA + miRNA -> C1A; K1 * kOff * mRNA * miRNA / V
JC1AR: C1A -> mRNA + miRNA; kOff * C1A
JC1ADI: C1A -> mRNA; k_i * b1 * C1A
JC1ADM: C1A -> miRNA; k_m * a1 * C1A
JC1AC2: C1A + miRNA -> C2; K2 * kOff * C1A * miRNA / V
JC1AC2R: C2 -> C1A + miRNA; kOff * C2
JC2DIA: C2 -> C1A; k_i * b2 * C2
JC1B: mRNA + miRNA -> C1B; K1 * kOff * mRNA * miRNA / V
JC1BR: C1B -> mRNA + miRNA; kOff * C1B
JC1BDI: C1B -> mRNA; k_i * b1 * C1B
JC1BDM: C1B -> miRNA; k_m * a1 * C1B
JC1BC2: C1B + miRNA -> C2; K2 * kOff * C1B * miRNA / V
JC1BC2R: C2 -> C1B + miRNA; kOff * C2
JC2DIB: C2 -> C1B; k_i * b2 * C2
JC2DM: C2 -> 2 miRNA; k_m * a2 * C2

s_i = 1; s_m = 0.3; s_m0 = 5.7
k_i = 0.25; k_m = 1
kOff = 100; K1 = 1000; K2 = 1000
a1 = 1; a2 = 12; b1 = 1; b2 = 4
V = 1.0

miRNA = 0.8; mRNA = 0.1; C1A = 0; C1B = 0; C2 = 0
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
for mi in np.linspace(0.001, 10, 20):
    for mr in np.linspace(0.001, 10, 20):
        runner.reset()
        runner['mRNA'] = mr
        runner['miRNA'] = mi
        result = runner.simulate(0, 200, 2001)
        extreme_index = findextreme(result['[mRNA]'][tcutoff:])
        this_extreme = result['[mRNA]'][extreme_index + tcutoff]
        if (args.ic == 'lo' and this_extreme < extreme_mrna) or (args.ic == 'hi' and this_extreme > extreme_mrna):
            extreme_ic = {f: result[f'[{f}]'][extreme_index + tcutoff] for f in runner.fs()}
            extreme_mrna = extreme_ic['mRNA']
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
    for s, species in enumerate(['miRNA', 'mRNA', 'C1A', 'C1B', 'C2']):
        data[s, :, c] = result[f'[{species}]'] / v
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('results', data=data)
        for param in runner.ps():
            hf.attrs[param] = runner[param]
        hf.attrs['_last'] = c
