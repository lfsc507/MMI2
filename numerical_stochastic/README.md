# Stochastic simulation code

Stochastic simulations were run and analyzed primarily in Julia (version 1.6.1).
This folder contains the following items:

* TOML files describe the versions of Julia packages used.
* The `sde` folder contains Julia scripts that run stochastic differential equation simulations, both additive and multiplicative.
* The `gillespie` folder contains a Python script that runs exact Gillespie simulations.
* `analysisutils.jl` contains Julia functions used for analyzing stochastic simulation results, e.g. quantifying bimodality.
