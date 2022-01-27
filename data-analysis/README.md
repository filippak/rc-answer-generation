# Analysis of the dataset
The purpose of this component is to give some quantiative measures of the available data. Specifically, we are interested in determinging how the correct answer and the question relate to the associated text. We want to investigate the following:
- In which sentence does in the associated paragraph does the answer appear?
- Of which word class is the sought answer?
- How "central" is the sentence in which the answer can be found?
    - Here, central is determined by ranking sentences by textrank

## Requirements 
To run the dependency parsing use python version 3.7 if running in anaconda and 3.8 otherwise
Guide to installing stanza can be found at: https://stanfordnlp.github.io/stanza/installation_usage.html