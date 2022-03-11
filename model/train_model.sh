# Train the C -> A classifier
python train.py ./data/tokenized_data ./results/model_CA_5_epochs_weighted_loss.pkl answer-extraction 3 5

# Train the C,A -> R classifier
python train.py ./data/tokenized_CAR_data ./results/model_CAR_5_epochs_weighted_loss.pkl sentence-extraction 3 5

# Train C -> R classifier
python train.py ./data/tokenized_CR_data ./results/model_CR_5_epochs.pkl sentence-extraction 3 5