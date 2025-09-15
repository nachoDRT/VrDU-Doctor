chmod +x src/train.py
chmod +x src/inference.py

python src/train.py \
    --dataset_name de-Rodrigo/merit \
    --dataset_subset es-digital-paragraph-degradation-seq \
    --freeze_encoder \
#     # --save_initial

# python src/inference.py \
#     --dataset_name de-Rodrigo/merit-secret \
#     --subset_name all \
#     --llava_model_version de-Rodrigo/llava-merit \
#     --subfolder llava_es-digital-seq
