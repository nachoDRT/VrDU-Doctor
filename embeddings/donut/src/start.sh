chmod +x src/embeddings.py

python src/embeddings.py \
    --debug True \
    --dataset de-Rodrigo/merit-aux \
    --subset  IIT-CDIP \
    --split train \
    --model vanilla \
    --max_samples 150 \
    --embedding_computation weighted \
    --weight_strength 0.5 \
    --check_img_embeddings
