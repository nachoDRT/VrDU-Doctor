FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the working directory
WORKDIR /app

RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

# Copy the 'transformers' library and set the environment variable so Python uses the modified library
COPY transformers /app/transformers
ENV PYTHONPATH=/app/transformers/src

# Copy and install requirements
COPY ablation/layoutxlm/dockerfiles/rtx2080/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY ablation/layoutxlm/ /app

# Execute the training
RUN chmod +x /app/src/start.sh
CMD ["bash", "/app/src/start.sh"]
