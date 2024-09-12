chmod +x src/train.py
chmod +x src/utils/inspect_hf_dataset.py

# python src/utils/inspect_hf_dataset.py

python src/train.py \
    --debug False \
    --dataset_name naver-clova-ix/cord-v2 \
    --dataset_subset ""

# python src/inference.py