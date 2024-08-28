# Nvidia RTX2080/3090/4090
Execute from ./VrDU-Doctor/single/chameleon

Available cards: `RTX2080`, `RTX3090`, `RTX4090`

### Create the docker :whale:
```bash
docker build -f dockerfiles/your_card/Dockerfile -t chameleon .
```

### Run :boom: or Debug :no_entry_sign::bug: the docker
```bash
docker run -p 5678:5678 -v /your_path_to_chameleon_files:/app/chameleon_files -v /your_path_to_data_files:/app/data-it --gpus all chameleon
```
