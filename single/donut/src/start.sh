chmod +x src/train.py
chmod +x src/utils/inspect_hf_dataset.py

# python src/utils/inspect_hf_dataset.py

python src/train.py \
    --debug True \
    --dataset_name cord-v2 \
    --dataset_subset ""

# python src/inference.py