# Nvidia 4090
Execute from ./VrDU-Doctor/single/paligemma

Available cards: `RTX4090`

### Create the docker :whale:
```bash
docker build -f dockerfiles/your_card/Dockerfile -t paligemma .
```

### Run :boom: or Debug :no_entry_sign::bug: the docker
```bash
docker run -it --gpus '"device=0"' --ipc=host paligemma
```

```bash
docker run -p 5678:5678 -it --gpus '"device=0"' -v "$HOME/.cache/huggingface":/root/.cache/huggingface --ipc=host paligemma
```