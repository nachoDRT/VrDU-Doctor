chmod +x src/embeddings.py

python src/embeddings.py \
    --debug True \
    --dataset de-Rodrigo/merit \
    --subset  es-digital-line-degradation-seq \
    --split train \
    --model vanilla \
    --max_samples 150