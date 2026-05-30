#!/bin/bash

# 1. 准备变量 (还是老一套)
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29503
NNODES=1
NODE_RANK=0
TOTAL_PROCESSES=$((NNODES * 8))

# 2. 清理环境
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_SIZE
unset RANK
unset WORLD_SIZE

# 3. 动态生成 FSDP 配置文件
# 注意：fsdp_transformer_layer_cls_to_wrap 建议填入你的 Transformer Block 类名
# 如果不知道，先空着，它会尝试自动策略，或者只做参数分片
CONFIG_FILE="fsdp_config_auto.yaml"

cat > $CONFIG_FILE <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
mixed_precision: bf16
num_machines: $NNODES
num_processes: $TOTAL_PROCESSES
machine_rank: $NODE_RANK
main_process_ip: $MASTER_ADDR
main_process_port: $MASTER_PORT
rdzv_backend: static
same_network: true
use_cpu: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
EOF

echo "=================================================="
echo "DEBUG: Switching to FSDP."
echo "DEBUG: Generated config for Node $NODE_RANK / $NNODES"
echo "=================================================="

# 4. 启动
accelerate launch \
    --config_file $CONFIG_FILE \
    train/train_nava.py \
	  --config configs/nava_mixtrain.yaml --resume Wan_5B.ckpt --load_ckpt_only
