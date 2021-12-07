# Python Gillespie simulation code

Gillespie simulations were run with Tellurium in CPython 3.8.5. All package versions are given in pip-compatible `requirements.txt`.
Each run of `gillespie.py` produces one HDF5 file containing a 3D array describing the timecourse of one population's simulation
(indexed by species, time index, and cell in Python, but note that Python's array dimension storage order is opposite Julia's).
The shell scripts in this folder launch Gillespie simulations of representative conditions, or resume the simulations if they were terminated unexpectedly.
