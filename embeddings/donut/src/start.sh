chmod +x src/embeddings.py

python src/embeddings.py \
    --debug True \
    --dataset - \
    --subset  - \
    --split - \
    --model vanilla \
    --max_samples 150 \
    --embedding_computation weighted \
    --weight_strength 0.5 \
    --loop
    # --check_img_embeddings

    # --dataset de-Rodrigo/merit-aux \
    # --subset  IIT-CDIP \
    # --split train \
    # --model vanilla \
    # --max_samples 150 \
    # --embedding_computation weighted \
    # --weight_strength 0.5
