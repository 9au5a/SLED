#!/bin/bash
#
#SBATCH --job-name=daic_classification_severity_new_labels
#SBATCH --output=/ukp-storage-1/kuczykowska/code/SLED/daic_classification_severity.log
#SBATCH --mail-user=paulina.kuczykowska@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2

source /ukp-storage-1/kuczykowska/code/SLED/sled_01/bin/activate
module load cuda/12.2
python3.9 /ukp-storage-1/kuczykowska/code/SLED/examples/seq2seq/run.py \
/ukp-storage-1/kuczykowska/code/SLED/examples/seq2seq/configs/data/daic_classification_severity.json \
/ukp-storage-1/kuczykowska/code/SLED/examples/seq2seq/configs/model/bart_base_sled.json \
/ukp-storage-1/kuczykowska/code/SLED/examples/seq2seq/configs/training/base_training_args.json \
--output_dir /ukp-storage-1/kuczykowska/code/SLED/tmp/daic_classification_severity_new_labels \
--learning_rate 2e-5 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \