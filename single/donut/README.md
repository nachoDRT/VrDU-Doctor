# Nvidia RTX2080
Execute from ./VrDU-Doctor/single/donut
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx2080/Dockerfile -t donut .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output donut
```

### Debug the docker :no_entry_sign::bug:
```bash
docker run -p 5678:5678 -it --gpus '"device=0"' -v /host_path_to_save_models:/app/models_output donut
```


# Remove *models* protected folder
Execute from ./VrDU-Doctor/single/donut
```bash
sudo rm -rf models
```