#!/bin/bash

# 获取脚本所在目录的绝对路径，并推导出项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# 分布式训练配置
NPROC_PER_NODE=1

# ========== PPO 训练参数 ==========
SAVE_DIR="../out"                      # 模型保存目录 (通过 ../out 映射到根目录的 out)
SAVE_WEIGHT="ppo_actor"                # 保存权重的前缀名
EPOCHS=1                               # 训练轮数
BATCH_SIZE=2                           # batch size
LEARNING_RATE="3e-7"                   # Actor 初始学习率
CRITIC_LEARNING_RATE="5e-7"            # Critic 初始学习率
DEVICE="cuda:0"                        # 训练设备
DTYPE="bfloat16"                       # 混合精度类型
NUM_WORKERS=8                          # 数据加载线程数
ACCUMULATION_STEPS=1                   # 梯度累积步数
GRAD_CLIP=1.0                          # 梯度裁剪阈值
LOG_INTERVAL=1                         # 日志打印间隔
SAVE_INTERVAL=10                       # 模型保存间隔

# ========== 模型与数据参数 ==========
HIDDEN_SIZE=768                        # 隐藏层维度
NUM_HIDDEN_LAYERS=8                    # 隐藏层数量
MAX_SEQ_LEN=768                        # Prompt最大长度
MAX_GEN_LEN=1024                       # 生成的最大长度
USE_MOE=1                              # 是否使用MoE架构（0=否，1=是）
DATA_PATH="../dataset/rlaif.jsonl"     # RLAIF数据路径
FROM_WEIGHT="full_sft"                 # 基于哪个权重训练 (不需要带后缀，脚本会自动拼接)
REWARD_MODEL_PATH="../../internlm2-1_8b-reward" # Reward 模型路径 (注意这里的相对路径)
FROM_RESUME=0                          # 是否自动检测&续训（0=否，1=是）

# ========== PPO 核心算法参数 ==========
CLIP_EPSILON=0.2                       # PPO裁剪参数
VF_COEF=0.5                            # Value function系数
KL_COEF=0.02                           # KL散度惩罚系数
GAMMA=1.0                              # GAE折扣因子
LAM=0.95                               # GAE lambda参数
CLIPRANGE_VALUE=0.2                    # Value function裁剪范围
PPO_UPDATE_ITERS=2                     # 同一批rollout重复更新次数
EARLY_STOP_KL=0.25                     # PPO early stop 的 KL 阈值
MINI_BATCH_SIZE=2                      # PPO每次更新的minibatch大小

# ========== Rollout 引擎参数 ==========
ROLLOUT_ENGINE="torch"                 # rollout引擎类型 (torch 或 sglang)
THINKING_RATIO=0.9                     # 按概率开启thinking
SGLANG_BASE_URL="http://localhost:8998"
SGLANG_MODEL_PATH="../model"
SGLANG_SHARED_PATH="./sglang_ckpt_ppo"

# ========== Wandb 与优化 ==========
USE_WANDB="--use_wandb"                # 是否使用wandb (如果不使用，可以注释掉此行或留空)
WANDB_PROJECT="MiniMind-PPO"           # wandb项目名
USE_COMPILE=0                          # 是否使用torch.compile加速（0=否，1=是）
DEBUG_MODE="--debug_mode"              # 是否打印训练调试采样
DEBUG_INTERVAL=20                      # debug模式下每隔多少step打印一次采样

# ==============================================================================

# 获取当前时间，格式为 YYYYMMDD_HHMMSS
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/self_test/outputs"
LOG_FILE="${LOG_DIR}/train_ppo_${CURRENT_TIME}.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

echo "🚀 开始启动 PPO (RLHF/RLAIF) 训练..."
echo "📂 切换到工作目录: $PROJECT_ROOT/trainer"
echo "📝 训练日志将输出到: $LOG_FILE"
echo "========================================="

# 切换到 trainer 目录执行，以兼容代码中写死的相对路径 (如 ../model 等)
cd "$PROJECT_ROOT/trainer"

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE train_ppo.py \
    --save_dir $SAVE_DIR \
    --save_weight $SAVE_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --critic_learning_rate $CRITIC_LEARNING_RATE \
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
    --max_gen_len $MAX_GEN_LEN \
    --use_moe $USE_MOE \
    --data_path $DATA_PATH \
    --from_weight $FROM_WEIGHT \
    --reward_model_path $REWARD_MODEL_PATH \
    --from_resume $FROM_RESUME \
    --clip_epsilon $CLIP_EPSILON \
    --vf_coef $VF_COEF \
    --kl_coef $KL_COEF \
    --gamma $GAMMA \
    --lam $LAM \
    --cliprange_value $CLIPRANGE_VALUE \
    --ppo_update_iters $PPO_UPDATE_ITERS \
    --early_stop_kl $EARLY_STOP_KL \
    --mini_batch_size $MINI_BATCH_SIZE \
    --rollout_engine $ROLLOUT_ENGINE \
    --thinking_ratio $THINKING_RATIO \
    --sglang_base_url $SGLANG_BASE_URL \
    --sglang_model_path $SGLANG_MODEL_PATH \
    --sglang_shared_path $SGLANG_SHARED_PATH \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --use_compile $USE_COMPILE \
    $DEBUG_MODE \
    --debug_interval $DEBUG_INTERVAL 2>&1 | tee "$LOG_FILE"
