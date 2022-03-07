# Train the C -> A classifier
python train.py ./data/tokenized_train_data_arr.pkl ./results/model_CA_5_epochs.pkl answer-extraction 3 5

# Train the C,A -> R classifier
python train.py ./data/tokenized_sentence_extraction_train_data_arr.pkl ./results/model_CAR_3_epochs.pkl sentence-extraction 3 3