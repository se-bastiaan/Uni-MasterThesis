#! /bin/bash
#
#SBATCH -J inpainting-transformer-validate
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sversteeg@science.ru.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=csedu
#SBATCH --mem=20G
#SBATCH -N 1 -n 4
#

source /ceph/csedu-scratch/project/sversteeg/venv/bin/activate
which python
echo "Image type: ${IMAGE_TYPE}"
echo "Max epochs: ${EPOCHS}"
echo "Attention type: ${ATTENTION_TYPE}"
python main.py --image_type ${IMAGE_TYPE} --load_checkpoint ${CHECKPOINT} --infer --max_epochs ${EPOCHS} --attention_type ${ATTENTION_TYPE} --dataset /ceph/csedu-scratch/project/sversteeg/mvtec-ad/ --output_path /ceph/csedu-scratch/project/sversteeg/output