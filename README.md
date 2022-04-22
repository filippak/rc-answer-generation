# Reading comprehesion answer generation
Master thesis project on automating answer extraction from Swedish text.

## Data
The dataset that is used in this project is the SweQUAD-MC dataset described in the paper [BERT-based distractor generation for Swedish reading comprehension questions using a small-scale dataset](https://arxiv.org/abs/2108.03973) with dataset available on [github](https://github.com/dkalpakchi/SweQUAD-MC) and additional data collected as part of a master thesis [A Method for Automatic Question Answering in Swedish based on BERT](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-286001)

## Data pre-processing
In order to prepare the data for analysis and for training models pre-processing is needed.
The script `sh parse_data.sh` in `data-analysis/parse_data` runs through the necessary pre-processing steps. The data can then be analyzed or further pre-processed with labels to be used to fine-tune BERT models.  

### Data pre-processing for BERT fine-tuning
Further pre-processing is necessary in order to make the data suiteble for model fine-tuning. To prepare the data for fine-tuning of the different models, run `sh create_datasets.sh` located in `data-analysis/create_dataset`. The scripts will first partition the data into training and validation sets, and then add the necessary markers and label the data for the models. The data is saved in `/data` in the same sub-directory.

### Data analysis scripts
The notebooks for data analysis are located in the folder `data-analysis/data_analysis`. Notebooks can be run and use the data saved during the pre-processing step.

## Docker
### To train model with docker do:

1. Move the training data into `dockerProject/data/CA`, `dockerProject/data/CRA` and `dockerProject/data/CAR_classification`
1. Create a folder `dockerProject/results`. This is where the trained model will be saved.
1. `cd dockerProject`
1. `docker build . -t rc_answer_extraction`
    1. This will build a docker image with the necessary dependencies to run the training on CPU
1. Run the docker container
    1. For older versions of nvidia and docker: `nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES=0 -v "$(pwd):/workspace" -v "$HOME/dockerProject/results:/workspace/results" rc_answer_extraction sh train_model.sh`
    1. For newer versions of nvidia and docker: `docker run --rm -t --shm-size=1g --gpus all -e CUBLAS_WORKSPACE_CONFIG=:16:8 -v "$(pwd):/workspace" -v "$HOME/dockerProject/results:/workspace/results" rc_answer_extraction sh train_model.sh`
    1. When setting the GPU: `docker run --rm -t --shm-size=1g --gpus '"device=0"' -e CUBLAS_WORKSPACE_CONFIG=:16:8 -v "$(pwd):/workspace" -v "$HOME/dockerProject/results:/workspace/results" karrfe_answer_extraction sh train_model.sh`

        1. `-e CUBLAS_WORKSPACE_CONFIG=:16:8` sets environment variable for reproducibility
        1. This will run the scripts specified in `train_model.sh` in the docker container on 1 GPU (0)
        1. `$HOME` has to be modified to fit the paths on the current system
        1. Optionally add argument `-d` to run the container in detached mode
        1. TODO: connect ports to use wandb.


### To transfer model to / from the server
If using an external server to train the model, the following command can be used to move files between server and local: `rsync -a <source path> <destination path>`

### Extra
The `helper.py` and `train.py` inside the `dockerProject` folder are modified to work for training the model using GPU resources, and will not work directly when using the CPU.

