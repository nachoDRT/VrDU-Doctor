# Nvidia RTX2080
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx2080/Dockerfile -t layoutxlm .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output layoutxlm
```

# Nvidia RTX3090
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx3090/Dockerfile -t layoutxlm .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output layoutxlm
```


# Nvidia RTX4090
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx4090/Dockerfile -t layoutxlm .
```

### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output layoutxlm
```

# Remove *models* protected folder
Execute from ./VrDU-Doctor/single/layoutlxlm
```bash
sudo rm -rf models
```