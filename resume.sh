#! /bin/bash
#
#SBATCH -J inpainting-transformer
#SBATCH --time=UNLIMITED
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sversteeg@science.ru.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=csedu
#SBATCH --mem=10G
#SBATCH -N 1 -n 4
#

source /ceph/csedu-scratch/project/sversteeg/venv/bin/activate
which python
echo "Image type: ${IMAGE_TYPE}"
echo "Attention type: ${ATTENTION_TYPE}"
echo "Checkpoint version: ${CHECKPOINT}"
python main.py --resume_checkpoint ${CHECKPOINT} --image_type ${IMAGE_TYPE} --max_epochs 20000 --attention_type ${ATTENTION_TYPE} --dataset /ceph/csedu-scratch/project/sversteeg/mvtec-ad/ --output_path /ceph/csedu-scratch/project/sversteeg/output