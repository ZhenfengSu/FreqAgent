##############Evaluation Parameters################
export MODEL_PATH=$1 # Model path

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch
export DATASET=$2 
export OUTPUT_PATH=$3 # Output path for prediction results

export TEMPERATURE=0.6 # LLM generation parameter, fixed at 0.6


######################################
### 1. Start server (background)   ###
######################################

# Activate server environment

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

# Optional dependency installation
# apt update
# apt install tmux -y
# pip install nvitop

echo "==== Starting Original Model SGLang Server (Port 6001)... ===="
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path $MODEL_PATH --host 0.0.0.0 --tp 2 --port 6001 &

ORIGINAL_SERVER_PID=$!