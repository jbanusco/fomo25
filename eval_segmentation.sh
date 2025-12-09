#!/bin/bash

# === Static config ===
SPLITS_FOLDER=/home/jovyan/shared/pedro-maciasgordaliza/fomo25/data/splits_final/task2/
SPLITS_FILENAME=${SPLITS_FOLDER}/nnunet_experiments/splits_final_no_test.json
DATA_DIR=/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data/fomo-task2/

# Local
SPLITS_FOLDER=/media/jaume/T7/data/splits_final/task2/
SPLITS_FILENAME=${SPLITS_FOLDER}/nnunet_experiments/splits_final_no_test.json
DATA_DIR=/media/jaume/T7/finetuning_data/fomo-task2/

# === Loop folds ===
for FOLD in 0 1 2 3 4; do
  echo "===================== FOLD ${FOLD} ====================="
    OUT_DIR=/media/jaume/T7/FOMO-MRI/model_test_jaume_6/Task002_FOMO2/mmunetvae/split_${FOLD}_aug_none_skip_ckpt/version_0/
    # OUT_DIR=/media/jaume/T7/FOMO-MRI/model_test_jaume_6/Task002_FOMO2/mmunetvae/split_${FOLD}_aug_all_skip_ckpt/version_0/
    MODEL_PATH=${OUT_DIR}/checkpoints/best_model.ckpt
        
    python /home/jaume/Desktop/Code/fomo25/src/predict.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUT_DIR}/predictions_eval_format \
    --task task2 \
    --checkpoint ${MODEL_PATH} \
    --which-split val \
    --split-idx ${FOLD}

    # Paths for evaluation
    GT_DIR=${DATA_DIR}/labels_flattened
    PRED_DIR="${OUT_DIR}/predictions_eval_format"
    SAVE_DIR="${OUT_DIR}/eval_results"
    mkdir -p "${SAVE_DIR}"

    echo "[FOLD ${FOLD}] Running evaluation..."
    python /home/jaume/Desktop/Code/container-validator/task2_segmentation/evaluation/seg_evaluator.py \
        "${GT_DIR}" \
        "${PRED_DIR}" \
        -l 0 1 \
        --output-dir "${SAVE_DIR}" \
        --verbose \
        --chill

    echo "[FOLD ${FOLD}] ✔ Done. Results -> ${SAVE_DIR}"
    
done

