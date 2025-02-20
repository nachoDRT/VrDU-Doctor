chmod +x src/embeddings.py

python src/embeddings.py \
    --dataset de-Rodrigo/merit-secret \
    --subset all \
    --model es-render-seq
    # --model es-digital-paragraph-degradation-seq
    # --model es-digital-line-degradation-seq
    # --model es-digital-seq
    # --model es-render-seq