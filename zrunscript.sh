#!/bin/bash -l
#SBATCH -J PyAnalys
#SBATCH -p epyc
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=72:00:00
#SBATCH -o zPyOutLog/pysc.out
#SBATCH -e zPyOutLog/pysc.err
#SBATCH --mail-user=ezhan039@ucr.edu
#SBATCH --mail-type=ALL

cd /rhome/ezhan039/bigdata/analysis/DualDestinies

python AllPhaseDiagrams.py
