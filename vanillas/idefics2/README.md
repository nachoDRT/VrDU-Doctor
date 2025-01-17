# Nvidia RTX2080
Execute from ./VrDU-Doctor/single/idefics2

Available cards: `RTX2080`

### Create the docker :whale:
```bash
docker build -f dockerfiles/your_card/Dockerfile -t idefics2 .
```

### Run :boom: or Debug :no_entry_sign::bug: the docker
```bash
docker run -p 5678:5678 -it --gpus all -v /host_path_to_save_models:/app/models_output --ipc=host idefics2 2>&1 | tee log.txt
```