# Nvidia RTX2080
Execute from ./VrDU-Doctor/bruteforce/layoutlxlm
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx2080_bruteforce/Dockerfile -t layoutxlm_bruteforce .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' layoutxlm_bruteforce
```

# Nvidia RTX3090:
Execute from ./VrDU-Doctor/bruteforce/layoutlxlm
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx3090_bruteforce/Dockerfile -t layoutxlm_bruteforce .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' layoutxlm_bruteforce
```


# Nvidia RTX4090:
Execute from ./VrDU-Doctor/bruteforce/layoutlxlm
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx4090_bruteforce/Dockerfile -t layoutxlm_bruteforce .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' layoutxlm_bruteforce
```