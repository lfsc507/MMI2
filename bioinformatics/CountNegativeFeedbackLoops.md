# Count of negative transcriptional feedback loops

Cycles of at most length 3 in the transcriptional [TRRUST2](https://www.grnpedia.org/trrust/) network were counted with [HiLoop](https://github.com/BenNordick/HiLoop):

```
(hiloop_pypy) $ python countmotifs.py trrust.gxml --maxcycle 3 --checknfl --nodecounts out/trrustnfl.csv
Finding cycles
Creating cycle intersection graphs
97 cycles, 519 node sharings
Searching for Type I and mixed-sign high-feedback motifs
Checking fused pairs
Searching for Type II and MISA motifs
PFL     45          
Type1   149
Type2   194
MISA    92
MISSA   36
uMISSA  10
MixHF   1544
Excite  254
```

Of the 97 cycles, 45 are positive, so the remaining 52 are negative.

The output table was then examined for genes involved in fewer positive feedback loops than total cycles (i.e. in at least one cycle that is not positive):

```
>>>> import pandas as pd
>>>> trrustnfl = pd.read_csv('out/trrustnfl.csv')
>>>> len(trrustnfl[trrustnfl['PFL'] < trrustnfl['Cycle']])
62
```

62 genes in the network are involved in at least one negative feedback loop of at most length 3.
