# Julia SDE simulation code

These scripts use the same Julia project/packages as specified in the parent directory.

* `runsim.jl`, `runsimrepressilator.jl`, and `runsimgeneosc.jl` simulate the MMI2-SSB, repressilator, and Kim *et al.* genetic oscillator models respectively.
They produce JLD2 data files representing a multilevel dictionary: the outermost dictionary contains an entry for each tested noise level,
the inner dictionaries contain an entry for each tested signal level,
and the innermost values are 3D arrays (indexed by molecular species, time index, and cell).
* `runsimcelldiv.jl` simulates the MMI2-SSB model similarly, but with a growing cell population.
Concentration values in the output's innermost arrays are NaN for cells that do not exist at that timepoint.
* `checkdims.jl` ensures that all population simulations within a data file reached the specified time index.
* `mergeresults.jl` copies the simulations of one data file into another data file,
used when a wider range of signal was subsequently tested or a simulation had to be rerun due to early termination.
