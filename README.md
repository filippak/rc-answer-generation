# Reading comprehesion answer generation
Master thesis project on automating answer extraction from Swedish text, as a step towards automatign the creation of reading comprehension questions.

## Data
The dataset that is used in this project is the SweQUAD-MC dataset described in the paper [BERT-based distractor generation for Swedish reading comprehension questions using a small-scale dataset](https://arxiv.org/abs/2108.03973) with dataset available on [github](https://github.com/dkalpakchi/SweQUAD-MC) and additional data collected as part of a master thesis [A Method for Automatic Question Answering in Swedish based on BERT](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-286001)

## Data pre-processing
In order to prepare the data for analysis and for training models pre-processing is needed.
The script `sh parse_data.sh` in `data_analysis/parse_data` runs through the necessary pre-processing steps. The data can then be analyzed or further pre-processed with labels to be used to fine-tune BERT models.  

### Data pre-processing for BERT fine-tuning
TODO

### Data analysis scripts
TODO

## Docker
To train model with docker do:

1. `cd dockerProject`
1. `docker build . -t rc_answer_extraction`
    1. This will build a docker image with the necessary dependencies to run the training on CPU
1. For older versions of nvidia and docker: `nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES=0 -v "$(pwd):/workspace" -v "$HOME/dockerProject/results:/workspace/results" rc_answer_extraction sh train_model.sh`
1. For newer versions of nvidia and docker: `docker run --rm --gpus all -v "$(pwd):/workspace" -v "$HOME/dockerProject/results:/workspace/results" rc_answer_extraction sh train_model.sh`
    1. This will run the scripts specified in `train_model.sh` in the docker container on 1 GPU (0)
    1. `$HOME` has to be modified to fit the paths on the current system
    1. Optionally add argument `-d` to run the container in detached mode
    1. TODO: connect ports to use wandb.

