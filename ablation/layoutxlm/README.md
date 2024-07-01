# Nvidia RTX2080
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f ablation/layoutxlm/dockerfiles/rtx2080/Dockerfile -t ablation .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output ablation
```

### Debug the docker :no_entry_sign::bug:
```bash
docker run -p 5678:5678 -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output ablation
```

**Note**: If you want to also debug files in the transformers library (imported as a submodule in the repo), you need to modify the ./VrDU-Doctor/ablation/layoutxlm/src/.vscode/launch.json file. Update your_local_path_to_project under "localRoot":
```json
"pathMappings": [
        {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "/app/src"
        },
        {
            "localRoot": "/your_local_path_to_project/VrDU-Doctor/transformers/src/transformers/models/layoutlmv2",
            "remoteRoot": "/app/transformers/src/transformers/models/layoutlmv2"
        }
    ]
```


# Nvidia RTX3090
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f ablation/layoutxlm/dockerfiles/rtx3090/Dockerfile -t ablation .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output ablation
```


# Nvidia RTX4090
Execute from ./VrDU-Doctor
### Create the docker :whale:
```bash
docker build -f ablation/layoutxlm/dockerfiles/rtx4090/Dockerfile -t ablation .
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