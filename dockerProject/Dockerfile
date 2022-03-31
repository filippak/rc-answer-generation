FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
# FROM huggingface/transformers-pytorch-gpu:4.17.0
# FROM 1.11.0-cuda11.3-cudnn8-runtime
WORKDIR /workspace
# copy everything from the current directory to teh docker container:
COPY . .
RUN apt-get install -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt