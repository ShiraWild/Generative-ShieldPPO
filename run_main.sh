#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### partition name where to run a job. Use ‘main’ unless qos is required. qos partitions ‘rtx3090’ ‘rtx2080’ ‘gtx1080’
#SBATCH --time 7-00:00:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name run_ppo ### name of the job. replace my_job with your desired job name
#SBATCH --output run_outputs/my_job-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=wilds@post.bgu.ac.il ### user’s email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=0 ### number of GPUs. Choosing type e.g.: #SBATCH --gpus=gtx_1080:1 , or rtx_2080, or rtx_3090 . Allocating more than 1 requires the IT team’s permission
##SBATCH --tasks=1 # 1 process – use for processing of few programs concurrently in a job (with srun). Use just 1 otherwise

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####

cd "/sise/home/wilds/Generative-ShieldPPO"
module load anaconda ### load anaconda module
source .venv/bin/activate ### activate a conda environment, replace my_env with your conda environment
python main.py --select_action_algo "ShieldPPO" --base_path "models/ShieldPPO_run/"