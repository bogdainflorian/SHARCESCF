#!/bin/bash

#$-N init__007_



PRIMARY_DIR=/home/fb445/Desktop/Sharc_water/DSPL_RESULTS/DSPL_007_p/

cd $PRIMARY_DIR
if [ -d ../../DSPL_RESULTS/DSPL_000_eq/SAVE ];
then
  if [ -d ./SAVE ];
  then
    rm -r ./SAVE
  fi
  cp -r ../../DSPL_RESULTS/DSPL_000_eq/SAVE ./
else
  echo "Should do a reference overlap calculation, but the reference data in ../../DSPL_RESULTS/DSPL_000_eq/ seems not OK."
  exit 1
fi

$SHARC/SHARC_RICC2.py QM.in >> QM.log 2>> QM.err
