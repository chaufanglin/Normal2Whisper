#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

# Your job commands go below here
# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load devtoolset/7
echo "============"
# Computations should be started with 'srun'. For example:
srun python -u mt_data_gen_rq1.py > /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/${SLURM_JOBID}.out