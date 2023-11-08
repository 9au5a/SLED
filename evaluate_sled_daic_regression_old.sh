#!/bin/bash
#
#SBATCH --job-name=daic_regression
#SBATCH --output=/ukp-storage-1/kuczykowska/code/SLED/R2_daic_evaluate2.log
#SBATCH --mail-user=paulina.kuczykowska@stud.tu-darmstadt.de
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2

source /ukp-storage-1/kuczykowska/code/SLED/sled_01/bin/activate
module load cuda/12.2
python3.9 /ukp-storage-1/kuczykowska/code/SLED/examples/seq2seq/evaluate_regression_daic.py