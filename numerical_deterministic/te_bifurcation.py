import tellurium as te
import re
import os
import numpy as np
import pandas as pd

def extract_data(r=None):
    '''
    Extract continuation data from fort.7 and fort.8 files in the current folder after an AUTO run
    r: tellurium object for obtaining variable names via r.fs()
    return:
        data: continuation curve in numpy array (r is not supplied) or pandas dataframe (r is supplied)
        bounds: indices of special points
        boundsh: indices of Hopf bifurcation points
    '''
    with open('fort.7', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines[12:]:
        if re.match(r'\s+0', line):
            break
        l = line.rstrip()
        fbs = [0, 3, 9, 13, 19]
        fs = []
        for w in range(1, len(fbs)):
            if l[w] != '-':
                fs.append(l[fbs[w-1]+1:fbs[w]+1])
            else:
                fs.append(l[fbs[w-1]+1:fbs[w]])
        fs.extend(re.split(r' +', l.strip()[17:]))
        data.append([float(f) for f in fs if f != ''])
    data = np.array(data)
    if len(data.shape) == 1:
        return [], [], []
    data = data[data[:,3]>0,:]
    idsn = np.where(data[:,1]<0)[0]
    idsp = np.where(data[:,1]>0)[0]

    bksn = np.where((idsn[1:]-idsn[:-1])>1)[0]
    bksp = np.where((idsp[1:]-idsp[:-1])>1)[0]
    bks = np.where((data[:,2]==2)|(data[:,2]==1))[0]
    hpts = np.where((data[:,2]==3))[0]

    bounds = [0]+list(bks)+[len(data)]

    boundsh = hpts

    with open('fort.8', 'r') as f:
        f_str = f.read()

    blks = re.split('\n +[12] +.*\n', f_str)
    half_blk = int(blks[0].count('\n')/2)
    numlines = [re.split("\n",blk) for blk in blks]
    numlines[0] = numlines[0][1:]
    states = [[float(num)
        for num in re.split(' +', "".join(lines[:]).strip())[1:]
        ] for lines in numlines]
    data8 = np.array(states)[:,:-20]

    if data8.shape[1] > data.shape[1]-6:
        data = np.hstack([data, data8[:,data.shape[1]-6:]])

    if r:
        data = pd.DataFrame(data=data, columns=['ST', 'PT', 'TY', 'LAB', 'PAR', 'L2-NORM']+r.fs())

    return data, bounds, boundsh

def model2te(model, ics={}):
    '''
    Construct Antimony string (for Tellurium) from a model dictionary
    '''
    model_str = '// Reactions\n\t'
    model_str += '\n\n\t' + 'J0: $S -> ' +  list(model['vars'].keys())[0] + '; 0.0\n\t'

    j = 1
    for i, var in enumerate(sorted(model['vars'], reverse=False)):
        if var in model['aux']:
            continue
        de = model['vars'][var]
        model_str += 'J'+ str(j) + ': -> ' + var + '; ' + de + '\n\t'
        j += 1

    model_str += '\n// Aux variables\n\t'

    if 'aux' in model:
        for var in model['aux']:
            model_str += var + ' := ' + model['vars'][var] + '\n\t'

    model_str += '\n// Species Init\n\t'

    for k, v in ics.items():
        model_str += k + ' = ' + str(round(v,4)) + '; '
    model_str += 'S = 0.0'

    model_str += '\n\n// Parameters\n\t'

    for k, v in model['pars'].items():
        model_str += k + ' = ' + str(v) + '; '


    if 'events' in model:
        for ev in model['events']:
            model_str += ev + '; ' + '\n\t'

    #print(model_str)
    #exit()
    r = te.loada(model_str)

    return r

def run_bf(r, auto, dirc="Positive", par="", lims=[0, 1],
        ds=0.001, dsmin=1E-5, dsmax=1, npr=2,
        pre_sim_dur=10, nmx=10000, if_plot=False):
    '''
    Run continuation with a Tellurium model
    '''
    if dirc.lower()[:3] == "pos" or dirc == "+":
        dirc = "Positive"
    elif dirc.lower()[:3] == "neg" or dirc == "-":
        dirc = "Negative"
    # Setup properties
    #auto = Plugin("tel_auto2000")
    auto.setProperty("SBML", r.getCurrentSBML())
    auto.setProperty("ScanDirection", dirc)
    auto.setProperty("PrincipalContinuationParameter", par)
    auto.setProperty("PreSimulation", "True")
    auto.setProperty("PreSimulationDuration", pre_sim_dur)
    auto.setProperty("RL0", lims[0])
    auto.setProperty("RL1", lims[1])
    auto.setProperty("NMX", nmx)
    auto.setProperty("NPR", npr)
    auto.setProperty("KeepTempFiles", True)
    auto.setProperty("DS", ds)
    auto.setProperty("DSMIN", dsmin)
    auto.setProperty("DSMAX", dsmax)
    try:
        auto.execute()
    except Exception as err:
        return [], [], []
    pts     = auto.BifurcationPoints
    if if_plot == True:
        lbl = auto.BifurcationLabels
        biData = auto.BifurcationData
        biData.plotBifurcationDiagram(pts, lbl)
    if not os.path.exists('fort.7'):
        return [], [], []
    else:
        data, bounds, boundsh = extract_data(r)
    #if os.path.exists('fort.7'):
        #os.remove('fort.7')
        #os.remove('fort.8')
    return data, bounds, boundsh

