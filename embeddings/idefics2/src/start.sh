chmod +x src/embeddings.py

python src/embeddings.py \
    --debug False \
    --dataset de-Rodrigo/merit \
    --subset es-digital-seq \
    --split train \
    --model es-render-seq \
    --max_samples 150