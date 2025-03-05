chmod +x src/train.py
chmod +x src/utils/inspect_hf_dataset.py
chmod +x src/inference.py

# python src/utils/inspect_hf_dataset.py

python src/train.py \
    --debug False \
    --dataset_name de-Rodrigo/merit-secret \
    --dataset_subset real

# python src/inference.py \
#     --dataset de-Rodrigo/merit-secret \
#     --subset all \
#     --model es-render-seq
    # --model es-digital-paragraph-degradation-seq
    # --model es-digital-line-degradation-seq
    # --model es-digital-seq
    # --model es-render-seq