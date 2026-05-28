#!/bin/bash

# ==============================================================================
# MiniMind 预训练启动脚本
# ==============================================================================

# 设置工作目录为 trainer 目录，这样脚本里的相对路径（如 ../model）就能正确解析了
cd "$(dirname "$0")/../trainer" || exit

# ------------------------------------------------------------------------------
# 可配置参数区
# ------------------------------------------------------------------------------

# 1. 目录与保存相关
SAVE_DIR="./out"                        # 模型保存目录
SAVE_WEIGHT="pretrain"                  # 保存权重的前缀名
SAVE_INTERVAL=1000                      # 模型保存间隔 (Step)

# 2. 训练超参数
EPOCHS=2                                # 训练轮数
BATCH_SIZE=32                           # 单卡 Batch Size
LEARNING_RATE=5e-4                      # 初始学习率
ACCUMULATION_STEPS=8                    # 梯度累积步数
GRAD_CLIP=1.0                           # 梯度裁剪阈值

# 3. 硬件与性能设置
DEVICE="cuda:0"                         # 训练设备 (单卡使用 cuda:0，多卡由 torchrun 控制)
DTYPE="bfloat16"                        # 混合精度类型 (可选 bfloat16 或 float16)
NUM_WORKERS=8                           # 数据加载线程数
USE_COMPILE=1                           # 是否使用 torch.compile 加速（0=否，1=是）

# 4. 模型结构参数
HIDDEN_SIZE=768                         # 隐藏层维度
NUM_HIDDEN_LAYERS=8                     # 隐藏层数量
MAX_SEQ_LEN=340                         # 最大序列长度
USE_MOE=1                               # 是否使用 MoE 架构（0=否，1=是）

# 5. 数据与续训
DATA_PATH="./dataset/pretrain_t2t_mini.jsonl"  # 预训练数据路径
FROM_WEIGHT="none"                      # 基于哪个权重训练 (none 表示从头开始)
FROM_RESUME=0                           # 是否自动检测并续训（0=否，1=是）

# 6. 日志与监控
LOG_INTERVAL=100                        # 日志打印间隔 (Step)
USE_WANDB="--use_wandb"                 # 是否使用 wandb (留空字符串 "" 则不使用)
WANDB_PROJECT="MiniMind-Pretrain"       # wandb 项目名称

# ------------------------------------------------------------------------------
# 执行训练命令
# ------------------------------------------------------------------------------

# 获取当前时间，格式为 YYYYMMDD_HHMMSS
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="../self_test/outputs"
LOG_FILE="${LOG_DIR}/pretrain_${CURRENT_TIME}.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

echo "🚀 开始执行预训练脚本..."
echo "📂 当前工作目录: $(pwd)"
echo "📝 训练日志将输出到: $LOG_FILE"

# 使用 stdbuf 取消缓冲，并通过 tee 将输出同时显示在终端并写入文件
stdbuf -oL -eL python3 train_pretrain.py \
    --save_dir "$SAVE_DIR" \
    --save_weight "$SAVE_WEIGHT" \
    --save_interval $SAVE_INTERVAL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --grad_clip $GRAD_CLIP \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --num_workers $NUM_WORKERS \
    --use_compile $USE_COMPILE \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --max_seq_len $MAX_SEQ_LEN \
    --use_moe $USE_MOE \
    --data_path "$DATA_PATH" \
    --from_weight "$FROM_WEIGHT" \
    --from_resume $FROM_RESUME \
    --log_interval $LOG_INTERVAL \
    --wandb_project "$WANDB_PROJECT" \
    $USE_WANDB 2>&1 | tee "$LOG_FILE"
