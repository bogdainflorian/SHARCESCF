#!/bin/bash

#$-N init_000_e



PRIMARY_DIR=/home/fb445/Desktop/Sharc_water/DSPL_RESULTS/DSPL_000_eq/

cd $PRIMARY_DIR


$SHARC/SHARC_RICC2.py QM.in >> QM.log 2>> QM.err
