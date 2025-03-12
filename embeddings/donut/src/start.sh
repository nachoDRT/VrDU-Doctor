chmod +x src/embeddings.py

python src/embeddings.py \
    --debug True \
    --dataset de-Rodrigo/merit-secret \
    --subset  all \
    --split test \
    --model vanilla \
    --max_samples 150 \
    --check_img_embeddings
