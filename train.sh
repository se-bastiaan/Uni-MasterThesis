#! /bin/bash
#
#SBATCH -J inpainting-transformer
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sversteeg@science.ru.nl
#SBATCH --gres=gpu:4
#SBATCH --partition=csedu
#SBATCH --mem=16G
#SBATCH -N 1 -n 1
#

source /ceph/csedu-scratch/project/sversteeg/venv/bin/activate
which python
python main.py --image_type ${IMAGE_TYPE} --max_epochs ${EPOCHS} --attention_type ${ATTENTION_TYPE}
