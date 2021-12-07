nohup python gillespie.py lo gillespie/snic_v${V}_lo_57.h5 s_m=0 V=$V &
nohup python gillespie.py lo gillespie/snic_v${V}_lo_60.h5 s_m=0.3 V=$V &
nohup python gillespie.py lo gillespie/snic_v${V}_lo_63.h5 s_m=0.6 V=$V &
nohup python gillespie.py hi gillespie/snic_v${V}_hi_57.h5 s_m=0 V=$V &
nohup python gillespie.py hi gillespie/snic_v${V}_hi_60.h5 s_m=0.3 V=$V &
nohup python gillespie.py hi gillespie/snic_v${V}_hi_63.h5 s_m=0.6 V=$V &
nohup python gillespie.py lo gillespie/sn_v${V}_lo_29.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=2.9 V=$V &
nohup python gillespie.py lo gillespie/sn_v${V}_lo_31.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=3.1 V=$V &
nohup python gillespie.py lo gillespie/sn_v${V}_lo_33.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=3.3 V=$V &
nohup python gillespie.py hi gillespie/sn_v${V}_hi_29.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=2.9 V=$V &
nohup python gillespie.py hi gillespie/sn_v${V}_hi_31.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=3.1 V=$V &
nohup python gillespie.py hi gillespie/sn_v${V}_hi_33.h5 s_m0=0 a2=4 b1=0.5 b2=0.1 k_i=2 s_m=3.3 V=$V &
nohup python gillespie.py lo gillespie/hopf_v${V}_lo_31.h5 s_m0=3.1 b2=7 s_m=0 V=$V &
nohup python gillespie.py lo gillespie/hopf_v${V}_lo_34.h5 s_m0=3.1 b2=7 s_m=0.3 V=$V &
nohup python gillespie.py lo gillespie/hopf_v${V}_lo_37.h5 s_m0=3.1 b2=7 s_m=0.6 V=$V &
nohup python gillespie.py hi gillespie/hopf_v${V}_hi_31.h5 s_m0=3.1 b2=7 s_m=0 V=$V &
nohup python gillespie.py hi gillespie/hopf_v${V}_hi_34.h5 s_m0=3.1 b2=7 s_m=0.3 V=$V &
nohup python gillespie.py hi gillespie/hopf_v${V}_hi_37.h5 s_m0=3.1 b2=7 s_m=0.6 V=$V &
