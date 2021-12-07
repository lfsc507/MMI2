PARAMS="s_m0=6.9 b2=3"

nohup python gillespie.py lo gillespie/sl_quiet_lo_72.h5 $PARAMS s_m=0.3 V=2000 &
nohup python gillespie.py lo gillespie/sl_quiet_lo_75.h5 $PARAMS s_m=0.6 V=2000 &
nohup python gillespie.py lo gillespie/sl_quiet_lo_78.h5 $PARAMS s_m=0.9 V=2000 &
nohup python gillespie.py lo gillespie/sl_med_lo_72.h5 $PARAMS s_m=0.3 V=1000 &
nohup python gillespie.py lo gillespie/sl_med_lo_75.h5 $PARAMS s_m=0.6 V=1000 &
nohup python gillespie.py lo gillespie/sl_med_lo_78.h5 $PARAMS s_m=0.9 V=1000 &
nohup python gillespie.py lo gillespie/sl_loud_lo_72.h5 $PARAMS s_m=0.3 V=500 &
nohup python gillespie.py lo gillespie/sl_loud_lo_75.h5 $PARAMS s_m=0.6 V=500 &
nohup python gillespie.py lo gillespie/sl_loud_lo_78.h5 $PARAMS s_m=0.9 V=500 &
nohup python gillespie.py lo gillespie/sl_vloud_lo_72.h5 $PARAMS s_m=0.3 V=75 &
nohup python gillespie.py lo gillespie/sl_vloud_lo_75.h5 $PARAMS s_m=0.6 V=75 &
nohup python gillespie.py lo gillespie/sl_vloud_lo_78.h5 $PARAMS s_m=0.9 V=75 &

nohup python gillespie.py hi gillespie/sl_quiet_hi_72.h5 $PARAMS s_m=0.3 V=2000 &
nohup python gillespie.py hi gillespie/sl_quiet_hi_75.h5 $PARAMS s_m=0.6 V=2000 &
nohup python gillespie.py hi gillespie/sl_quiet_hi_78.h5 $PARAMS s_m=0.9 V=2000 &
nohup python gillespie.py hi gillespie/sl_med_hi_72.h5 $PARAMS s_m=0.3 V=1000 &
nohup python gillespie.py hi gillespie/sl_med_hi_75.h5 $PARAMS s_m=0.6 V=1000 &
nohup python gillespie.py hi gillespie/sl_med_hi_78.h5 $PARAMS s_m=0.9 V=1000 &
nohup python gillespie.py hi gillespie/sl_loud_hi_72.h5 $PARAMS s_m=0.3 V=500 &
nohup python gillespie.py hi gillespie/sl_loud_hi_75.h5 $PARAMS s_m=0.6 V=500 &
nohup python gillespie.py hi gillespie/sl_loud_hi_78.h5 $PARAMS s_m=0.9 V=500 &
nohup python gillespie.py hi gillespie/sl_vloud_hi_72.h5 $PARAMS s_m=0.3 V=75 &
nohup python gillespie.py hi gillespie/sl_vloud_hi_75.h5 $PARAMS s_m=0.6 V=75 &
nohup python gillespie.py hi gillespie/sl_vloud_hi_78.h5 $PARAMS s_m=0.9 V=75 &