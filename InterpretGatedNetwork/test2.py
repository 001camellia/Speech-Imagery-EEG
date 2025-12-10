#!/bin/bash
# eeg_runner_fixed.sh

# 数据参数
DATA=EEG3
DATA_ROOT="/root/autodl-tmp/InterpretGatedNetwork-main/data/preprocessed_fif"
JSON_PATH="/root/autodl-tmp/InterpretGatedNetwork-main/json/textmaps.json"
DATASET="EEG_Imagine_3Class"
BATCH_SIZE=8
MAX_FILES=5            
MAX_EPOCHS=10              

# 模型参数
MODEL=InterpGN
DNN_TYPE=FCN
NUM_SHAPELET=5
LAMBDA_DIV=0.1
LAMBDA_REG=0.1
EPS=1
BETA_SCHEDULE=constant
GATING_VALUE=1

# 模型输入维度
ENC_IN=122
C_OUT=3
SUBJECT_IDS="sub-01,sub-02,sub-03"
# 随机种子
SEEDS=(0 )

# 创建输出目录
mkdir -p /root/autodl-tmp/InterpretGatedNetwork-main/result

echo "============================================"
echo "EEG 想象任务 - 3分类模型训练"
echo "============================================"
echo "数据: $DATA"
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "最大文件数: $MAX_FILES"
echo "训练轮数: $MAX_EPOCHS"
echo "输入维度: $ENC_IN"
echo "输出类别: $C_OUT"
echo "JSON路径: $JSON_PATH"
echo "数据目录: $DATA_ROOT"
echo "============================================"

# 检查文件是否存在
echo "检查文件..."
if [[ -f "$JSON_PATH" ]]; then
    echo "✓ JSON文件存在: $JSON_PATH"
    echo "  JSON文件大小: $(du -h "$JSON_PATH" | cut -f1)"
    echo "  JSON文件行数: $(wc -l < "$JSON_PATH")"
else
    echo "✗ JSON文件不存在: $JSON_PATH"
    echo "查找json文件..."
    find /root/autodl-tmp -name "textmaps.json" 2>/dev/null
    exit 1
fi

if [[ -d "$DATA_ROOT" ]]; then
    echo "✓ 数据目录存在: $DATA_ROOT"
    echo "  数据目录内容:"
    ls -la "$DATA_ROOT" | head -10
else
    echo "✗ 数据目录不存在: $DATA_ROOT"
    exit 1
fi

cd /root/autodl-tmp/InterpretGatedNetwork-main

for seed in ${SEEDS[@]}; do
    echo -e "\n===================================================================="
    echo "随机种子: $seed"
    echo "===================================================================="
    
    # 移除所有行内注释！
    python run.py \
        --data "$DATA" \
        --data_root "$DATA_ROOT" \
        --json_path "$JSON_PATH" \
        --dataset "$DATASET" \
        --max_files "$MAX_FILES" \
        --enc_in "$ENC_IN" \
        --c_out "$C_OUT" \
        --model "$MODEL" \
        --dnn_type "$DNN_TYPE" \
        --num_shapelet "$NUM_SHAPELET" \
        --lambda_div "$LAMBDA_DIV" \
        --lambda_reg "$LAMBDA_REG" \
        --epsilon "$EPS" \
        --beta_schedule "$BETA_SCHEDULE" \
        --gating_value "$GATING_VALUE" \
        --train_epochs "$MAX_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 5e-3 \
        --dropout 0. \
        --seed "$seed" \
        --amp \
        --task_name classification 
done


