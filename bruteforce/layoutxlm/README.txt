# Execute from ./VrDU-Doctor/bruteforce/layoutlxlm


IF YOU WANT TO RUN A BRUTEFORCE TRAINING SESSION WITH 'N' SUBSETS COMBINATIONS DATASET

# For Nvidia RTX2080:

# Create the docker
docker build -f dockerfiles/rtx2080_bruteforce/Dockerfile -t layoutxlm_bruteforce .

# For Nvidia RTX3090:

# Create the docker
docker build -f dockerfiles/rtx3090_bruteforce/Dockerfile -t layoutxlm_bruteforce .


# For Nvidia RTX4090:

# Create the docker
docker build -f dockerfiles/rtx4090_bruteforce/Dockerfile -t layoutxlm_bruteforce .

# Run the docker in your GPU:
docker run -it --gpus '"device=0"' layoutxlm_bruteforce
