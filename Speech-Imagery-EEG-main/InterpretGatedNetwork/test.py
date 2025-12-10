#!/bin/bash

# Model Configs, default settings, change as needed
MODEL=InterpGN
DNN_TYPE=FCN
NUM_SHAPELET=5
LAMBDA_DIV=0.1
LAMBDA_REG=0.1
EPS=1
BETA_SCHEDULE=constant
GATING_VALUE=1


'''UEA_DATASETS=(
    "UWaveGestureLibrary"
)'''

UEA_DATASETS=(
    "sub-01"
)

SEEDS=(0 42 )

for dataset in ${UEA_DATASETS[@]}; do
    for seed in ${SEEDS[@]}; do
        python run.py \
            --model $MODEL \
            --dnn_type $DNN_TYPE \
            --dataset $dataset \
            --train_epochs 5 \
            --batch_size 1 \
            --lr 5e-3 \
            --dropout 0. \
            --num_shapelet $NUM_SHAPELET \
            --lambda_div $LAMBDA_DIV \
            --lambda_reg $LAMBDA_REG \
            --epsilon $EPS \
            --beta_schedule $BETA_SCHEDULE \
            --seed $seed \
            --gating_value $GATING_VALUE \
            --amp
    done
done