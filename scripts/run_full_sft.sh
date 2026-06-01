#!/bin/bash

# 获取脚本所在目录的绝对路径，并推导出项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

# 分布式训练配置
# 如果是单卡机器，可以直接使用 python trainer/train_full_sft.py，不使用 torchrun
NPROC_PER_NODE=1

# ========== 全量微调训练参数 ==========
SAVE_DIR="../trainer/out"              # 模型保存目录 (通过 ../trainer/out 指定为绝对相对路径)
SAVE_WEIGHT="full_sft"                 # 保存权重的前缀名
EPOCHS=2                               # 训练轮数
BATCH_SIZE=16                          # batch size
LEARNING_RATE="1e-5"                   # 初始学习率
DEVICE="cuda:0"                        # 训练设备
DTYPE="bfloat16"                       # 混合精度类型 (bfloat16 或 float16)
NUM_WORKERS=8                          # 数据加载线程数
ACCUMULATION_STEPS=1                   # 梯度累积步数
GRAD_CLIP=1.0                          # 梯度裁剪阈值
LOG_INTERVAL=100                       # 日志打印间隔
SAVE_INTERVAL=1000                     # 模型保存间隔

# ========== 模型与数据参数 ==========
HIDDEN_SIZE=768                        # 隐藏层维度
NUM_HIDDEN_LAYERS=8                    # 隐藏层数量
MAX_SEQ_LEN=768                        # 训练的最大截断长度
USE_MOE=1                              # 是否使用MoE架构（0=否，1=是）
DATA_PATH="../dataset/sft_t2t_mini.jsonl"  # 训练数据路径
FROM_WEIGHT="pretrain"                 # 基于哪个权重训练 (不需要带 _768_moe.pth 后缀，脚本会自动拼接)
FROM_RESUME=0                          # 是否自动检测&续训（0=否，1=是）

# ========== Wandb 与优化 ==========
USE_WANDB="--use_wandb"                # 是否使用wandb (如果不使用，可以注释掉此行或留空)
WANDB_PROJECT="MiniMind-Full-SFT"      # wandb项目名
USE_COMPILE=0                          # 是否使用torch.compile加速（0=否，1=是）


# 获取当前时间，格式为 YYYYMMDD_HHMMSS
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="../self_test/outputs"
LOG_FILE="${LOG_DIR}/full_sft_${CURRENT_TIME}.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

echo "🚀 开始启动全量微调 (Full SFT) 训练..."
echo "📂 切换到工作目录: $PROJECT_ROOT/trainer"
echo "📝 训练日志将输出到: $LOG_FILE"
echo "========================================="

# 切换到 trainer 目录执行，以兼容代码中写死的相对路径 (如 ../model 等)
cd "$PROJECT_ROOT/trainer"

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE train_full_sft.py \
    --save_dir $SAVE_DIR \
    --save_weight $SAVE_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --dtype $DTYPE \
    --num_workers $NUM_WORKERS \
    --accumulation_steps $ACCUMULATION_STEPS \
    --grad_clip $GRAD_CLIP \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --max_seq_len $MAX_SEQ_LEN \
    --use_moe $USE_MOE \
    --data_path $DATA_PATH \
    --from_weight $FROM_WEIGHT \
    --from_resume $FROM_RESUME \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --use_compile $USE_COMPILE 2>&1 | tee "$LOG_FILE"
