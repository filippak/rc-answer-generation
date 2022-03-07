import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset

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

