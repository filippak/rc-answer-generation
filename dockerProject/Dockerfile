# FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime <-- OLD
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
WORKDIR /workspace
# copy everything from the current directory to teh docker container. Not needed if using the -v flag
# COPY . .
COPY  requirements.txt /tmp/
RUN apt-get install -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt