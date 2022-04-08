
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
 # weights are calculated in the create_CA_dataset.ipynb

class WeightedLossTrainerCA(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # INS: [0.51, 72.43, 24.36]
        # ISNS [0.717, 8.511, 4.936]
        # INS15  [0.64, 17.38, 8.40]
        # INS125 [0.58, 30.76, 12.86]
        # INS1375 [0.61, 22.53, 10.20]
        # INS1125 [0.55, 45.01, 17.08]

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.64, 17.38, 8.40]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCAR(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # ISNS weights: [0.74, 10.49, 2.54]
        # INS weights: [0.54, 110.06, 6.45]

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.74, 10.49, 2.54]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCRA(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # ISNS weights: [0.71, 17.93, 10.43]
        # INS weights: [ 0.50, 289.27, 98.10]
        # INS15 [0.63, 43.71, 21.27]
        # TODO: INS175 [0.67, 25.50, 13.74]
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.63, 43.71, 21.27]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class WeightedLossTrainerCARSentClass(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # INS: [0.54, 6.27] - too many 1's
        # ISNS:  [0.74, 2.50] - too many 0's
        # INS15 [0.66, 3.40]

        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.66, 3.40])) 
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


