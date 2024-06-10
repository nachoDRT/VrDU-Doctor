# Analytics 🧑‍💻

### Introduction
In this folder, you can obtain findings from your trained models. Again, the easiest way to proceed is by using a dockerized solution.

# Nvidia RTX2080
Execute from ./VrDU-Doctor/analytics
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx2080/Dockerfile -t analysis .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' analysis
```
### Debug the docker :no_entry_sign::bug:
```bash
docker run -p 5678:5678 -it --gpus '"device=0"' analysis
```

# Nvidia RTX3090
Execute from ./VrDU-Doctor/analytics
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx3090/Dockerfile -t analysis .
```
### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v analysis
```


# Nvidia RTX4090
Execute from ./VrDU-Doctor/analytics
### Create the docker :whale:
```bash
docker build -f dockerfiles/rtx4090/Dockerfile -t analysis .
```

### Run the docker in your GPU :boom:
```bash
docker run -it --gpus '"device=0"' -v analysis
```