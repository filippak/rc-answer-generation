
import torch
from torch.utils.data import Dataset
from torch import nn
from transformers import Trainer

# Make a Dataset class for the data following instructions: 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://huggingface.co/transformers/v3.2.0/custom_datasets.html

class ContextAnswerDataset(Dataset):
    """
    Class represeting the labeled dataset for answer extraction.
    """

    def __init__(self, data_arr, transform=None):
        self.context_answer_arr = data_arr
        self.transform = transform
    
    def __len__(self):
        return len(self.context_answer_arr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.context_answer_arr[idx]

        if self.transform:
            sample = self.transform(sample)
        
        return sample


# Make a custom trainer to use a weighted loss
# https://huggingface.co/docs/transformers/main_classes/trainer#:%7E:text=passed%20at%20init.-,compute_loss,-%2D%20Computes%20the%20loss

class WeightedLossTrainerCA(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # TODO: try with different loss functions!
        # weights are retrieved in the create_CA_dataset.ipynb by weighting the labels inversely by how often they occur
        # INS: [0.51, 72.43, 24.36]
        # ISNS [0.717, 8.511, 4.936]

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.51, 72.43, 24.36]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCAR(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # TODO: try with different loss functions!
        # weights are retrieved in the create_CA_dataset.ipynb by weighting the labels inversely by how often they occur
        # ISNS weights: [0.74, 10.49, 2.54]
        # INS weights: [0.54, 110.06, 6.45]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.74, 10.49, 2.54]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCRA(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # TODO: try with different loss functions!
        # weights are retrieved in the create_CA_dataset.ipynb by weighting the labels inversely by how often they occur
        # ISNS weights: [0.71,17.93, 10.43]
        # INS weights: [ 0.50, 321.60, 108.70]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.71, 17.93, 10.43]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCARSentClass(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # ISNS: [0.5, 6] - too many 1's
        # Manual Weights: [0.5, 2] - too many 0's
        # ISNS:  [0.455, 1.545] - too many 0's
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.455, 1.545])) 
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


