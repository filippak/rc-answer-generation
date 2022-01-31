# Reading comprehesion answer generation
Master thesis project on automating answer extraction from Swedish text, as a step towards automatign the creation of reading comprehension questions.

## Data
The dataset that is used in this project is the SweQUAD-MC dataset described in the paper [BERT-based distractor generation for Swedish reading comprehension questions using a small-scale dataset](https://arxiv.org/abs/2108.03973) with dataset available on [github](https://github.com/dkalpakchi/SweQUAD-MC).

## Running the scripts
Multiple different analysis are done on the data. To not have to re-run the data processing steps every time, the scripts save the processed data as pandas dataframes locally. Some scripts need other to have generated data before, and there is therfore an order to run the scripts in.

1. `load-data.ipynb`
1. `dependency_parsing.ipynb`
1. order is independent for the following scripts 
    1. `answer_location_statistics.ipynb`
    1. `text_rank.ipynb`
    1. `word_overlap.ipynb`
    1. `interrogative_words.ipynb`  
