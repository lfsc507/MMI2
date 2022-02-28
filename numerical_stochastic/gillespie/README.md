# Python Gillespie simulation code

Gillespie simulations were run with Tellurium in CPython 3.8.5. All package versions are given in pip-compatible `requirements.txt`.
Each run of a Python script here produces one HDF5 file containing a 3D array describing the timecourse of one population's simulation
(indexed by species, time index, and cell in Python, but note that Python's array dimension storage order is opposite Julia's).
`gillespie.py` simulates the MMI2-SSB Model; `gillespie_geneosc.py` simulates the Kim *et al.* genetic oscillator model.

The shell scripts in this folder launch Gillespie simulations of representative conditions, or resume the simulations if they were terminated unexpectedly.
`startgillespienoise.sh` and `startgillespiegeneosc.sh` expect a `V` environment variable specifying the cell volume.
