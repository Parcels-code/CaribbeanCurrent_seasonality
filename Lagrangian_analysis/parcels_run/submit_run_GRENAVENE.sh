#!/bin/bash
#SBATCH -J GRcr050            # name of the job
#SBATCH -p normal             # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 16                 # number of cores
#SBATCH -t 5-00:00:00         # number of hours you want to reserve the cores
#SBATCH -o logfiles/run_GRENAVENE_coastrep050ED.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/run_GRENAVENE_coastrep050ED.err     # name of the error file (if there are errors)


module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

now="$(date)"
printf "Start date and time %s\n" "$now"

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/CaribbeanCurrent_seasonality/Lagrangian_analysis/parcels_run/

conda activate parcels-dev-local

s_time=$(date +%s)

batch_nr=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")

for i in "${batch_nr[@]}"; do
    echo "Running batch $i"
    python 3_run_GRENAVENE.py $i > logfiles/run_GRENAVENE_coastrep050ED_$i.log 2>&1 &
done

wait

e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time )) seconds"