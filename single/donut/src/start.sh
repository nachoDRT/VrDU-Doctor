chmod +x src/train.py
chmod +x src/utils/inspect_hf_dataset.py

# python src/utils/inspect_hf_dataset.py

python src/train.py \
    --debug False \
    --dataset_name de-Rodrigo/merit \
    --dataset_subset 'en-digital-seq'

# python src/inference.py