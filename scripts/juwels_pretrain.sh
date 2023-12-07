#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=bakllava-aurora-lumi-dev
#SBATCH --account=cstdl
#SBATCH --partition=booster
#SBATCH --time=2:00:00
#SBATCH --threads-per-core=1
#SBATCH --output /p/project/ccstdl/raj3/aurora-m/slurm-output/bakllava-runs-dev-%j.out

CONDA_ENV="/p/project/ccstdl/shared/generic/miniconda3/envs/raj3_bakklava"
MINICONDA_PATH="/p/project/ccstdl/shared/generic/miniconda3"


MODEL_VERSION="aurora-m/Aurora-40k-hf"
EXP_NAME="bakllava-quadrant-$MODEL_VERSION-pretrain-save_all"

# BAKLLAVA_PATH="/p/project/laionize/marianna/bakllava_original/BakLLaVA"
BAKLLAVA_PATH="/p/project/ccstdl/raj3/aurora-m"

DATA_PATH="/p/scratch/ccstdl/marianna/bakllava/blip_laion_cc_sbu_558k.json"
TEXT_DATA_PATH="/p/scratch/ccstdl/lumi-data/en/books_2k_part_aa.jsonl"
IMAGE_FOLDER="/p/scratch/ccstdl/marianna/bakllava/images.zip"
VISION_TOWER="openai/clip-vit-large-patch14"
OUTPUT_DIR="/p/scratch/ccstdl/raj3/aurora-m/checkpoints/$EXP_NAME"

source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Allow communication over InfiniBand cells.

export MASTER_ADDR="${MASTER_ADDR}.juwels"

export MASTER_PORT=6000
export NUM_NODES=8
export NUM_GPUS=4
# export HOSTFILE_PATH="/p/project/laionize/marianna/bakllava_original/hostfile1"
export NCCL_DEBUG=INFO

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

export PYTHONPATH="$PYTHONPATH:${BAKLLAVA_PATH}"



cd ${BAKLLAVA_PATH}

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`i: \
    --tee 3 \
    "

export CMD="llava/train/train_mem.py \
    --train_supervised False \
    --model_name_or_path $MODEL_VERSION \
    --cache_dir "/p/scratch/ccstdl/raj3" \
    --version plain \
    --dataset_type "files" \
    --image_folder $IMAGE_FOLDER \
    --data_path $DATA_PATH \
    --unsupervised_data_path $TEXT_DATA_PATH \
    --vision_tower $VISION_TOWER \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --resume_from_checkpoint False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --deepspeed ./scripts/zero2.json "

srun --wait=60 \
    --kill-on-bad-exit=1 \
     --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role ${SLURMD_NODENAME}i: $CMD"
