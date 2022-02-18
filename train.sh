#! /bin/bash
#
#SBATCH -e slurm-%j-${IMAGE_TYPE}-${EPOCHS}-${ATTENTION_TYPE}.out
#SBATCH -o slurm-%j-${IMAGE_TYPE}-${EPOCHS}-${ATTENTION_TYPE}.out
#SBATCH -J inpainting-transformer
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sversteeg@science.ru.nl
#SBATCH --gres=gpu:4
#SBATCH --partition=csedu
#SBATCH --mem=16G
#SBATCH -N 1 -n 4
#

source /ceph/csedu-scratch/project/sversteeg/venv/bin/activate
which python
echo "Image type: ${IMAGE_TYPE}"
echo "Max epochs: ${EPOCHS}"
echo "Attention type: ${ATTENTION_TYPE}"
python main.py --image_type ${IMAGE_TYPE} --max_epochs ${EPOCHS} --attention_type ${ATTENTION_TYPE} --dataset /ceph/csedu-scratch/project/sversteeg/mvtec-ad/ --output__path /ceph/csedu-scratch/project/sversteeg/output
