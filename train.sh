#! /bin/bash
#
#SBATCH -J inpainting-transformer
#SBATCH --time=INFINITE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sversteeg@science.ru.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=csedu
#SBATCH --mem=60G
#SBATCH -N 1 -n 4
#

source /ceph/csedu-scratch/project/sversteeg/venv/bin/activate
which python
echo "Image type: ${IMAGE_TYPE}"
echo "Attention type: ${ATTENTION_TYPE}"
python main.py --image_type ${IMAGE_TYPE} --batch_size 256 --patience 150 --max_epochs ${EPOCHS} --attention_type ${ATTENTION_TYPE} --dataset /ceph/csedu-scratch/project/sversteeg/mvtec-ad/ --output_path /ceph/csedu-scratch/project/sversteeg/output