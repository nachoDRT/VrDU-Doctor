chmod +x src/train.py
chmod +x src/inference.py

python src/train.py \
    --debug False \
    --dataset de-Rodrigo/merit \
    --subset es-digital-seq \
    --freeze_encoder \
    --test_real \
    --train_combination \
    --combination_info de-Rodrigo/merit,es-digital-seq,britanico,test \
    # --save_initial


# python src/inference.py \
#     --dataset de-Rodrigo/merit-secret \
#     --subset all \
#     --model es-digital-noisy-degradation-seq