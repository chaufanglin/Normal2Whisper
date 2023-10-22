#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL

# Your job commands go below here
# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module load devtoolset/7
echo "============"
# Computations should be started with 'srun'. For example:
srun python -u mt_data_gen_rq1_500.py > /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/Normal2Whisper/${SLURM_JOBID}.out
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/aa --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ab --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ac --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ad --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ae --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/af --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ag --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out & 
# srun --exclusive --ntasks=1 python -u data_gen.py --data_list ./500_split/ah --output_dir /tudelft.net/staff-bulk/ewi/insy/SpeechLab/zhaofenglin/LibriSpeech/train-other-500-pw >> ./${SLURM_JOBID}.out &
# wait