How to use the ESCF.py script:
Date: 16.12.2024
1) set up all LVC directories using the Sharc tools. For this you will need the mos file from ridft or dscf and a molden.input file. Additionally a template file is required
2) template file musst be named ESCF.template and may include the following keywords:

basis all *basis set* 
OR
basis '*ATOM*' *basis set* '*ATOM*' *basis set* '*ATOM*' *basis set* ... <--- The basis sets musst be availabe in Turbomole

auxbasis all *basis set* 
OR
auxbasis '*ATOM*' *basis set* '*ATOM*' *basis set* '*ATOM*' *basis set* ...  <--- jbas or cbas

charge *Chargenumber*

rpacor *Integer in MB* <--- how much RAM

gw <--- if you want GW-BSE instead of TD-DFT

functional *functional* <--- musst be available in Turbomole
OR
functional LC-blyp alpha beta omega


qpeiter *integer or evGW* <--- to later for self-consistency

QP_energies *start* *end* <--- orbital range for contour deformations in GW

tda <--- to use TDA and not RPA

grid *gridsize* <--- musst be available in Turbomole

ncore <--- to limit number of determinants for the calculations of overlaps

scfconv *integer* <--- how many integers are required for scf convergence

3) the resource file musst be named ESCF.resources, but the SHARC RICC2 format can be used.

4) make sure SHARC and Turbomole are loaded (in the PATH) before executing. (The code will check this and stop otherwise)

5) I used the following slurm file for each single point:

SCRIPT_DIR=/path/to/ESCF.py
PRIMARY_DIR=$(pwd)
cd $PRIMARY_DIR

$SCRIPT_DIR/ESCF.py QM.in >> QM.log 2>> QM.err
