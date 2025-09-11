chmod +x src/embeddings.py

python src/embeddings.py \
    --dataset de-Rodrigo/merit-secret \
    --subset  all \
    --split test \
    --model vanilla \
    --max_samples 150 \
    # --loop \
    # --debug