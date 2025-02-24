chmod +x src/train.py
chmod +x src/inference.py

# python src/train.py \
#     --debug False \
#     --dataset de-Rodrigo/merit \
#     --subset es-digital-rotation-zoom-degradation-seq

python src/inference.py \
    --dataset de-Rodrigo/merit-secret \
    --subset all \
    --model es-digital-rotation-zoom-degradation-seq