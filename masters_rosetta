#!/bin/bash

#SBATCH --job-name=afrelaxrununcut
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --array=1-43

echo 'Run uncut Job running'

shopt -s extglob

echo 'Setting S variable'
S=$(head -${SLURM_ARRAY_TASK_ID} dhrlist | tail -1)

echo 'Running rosetta_scripts command'
srun rosetta_scripts.default.linuxgccrelease -s './dimer_results_withmsa_2chain/D'`basename $S`'AB_real_'*'.result/D'`basename $S`'AB_real_'*'_relaxed_rank_'?'_model_'?'.pdb' -nstruct 1 -in:file:native './5rep_DHRs_dimers/D'`basename $S`'AB_real_'*'.pdb' -parser:protocol ./relax_AFstructs_wconstraints.xml -ex1 -ex2 -out:path:all './dimer_results_withmsa_2chain/D'`basename $S`'AB_real_'*'.result/' -out:suffix '_AFRMSD' -overwrite > ./D`basename $S`/log_relax_af

echo 'Finished running rosetta_scripts'

echo 'pdbname='$S
