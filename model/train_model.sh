# Train the C -> A classifier, INS weights
python train.py ./data/CA/tokenized_CA_data ./results/model_CA_3e_ISNS_weighted_loss.pkl answer-extraction 3 3

# Train the C -> A classifier, ISNS weights
python train.py ./data/CA/tokenized_CA_data ./results/model_CA_3e_INS_weighted_loss.pkl answer-extraction 3 3

# Train the C,A -> R classifier
python train.py ./data/CAR/tokenized_CAR_data ./results/model_CAR_3e_INS_weighted_loss.pkl sentence-extraction 3 3 --CAR

# Train the C,R -> A classifier
python train.py ./data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_INS_weighted_loss.pkl answer-extraction-2 3 3 --CRA 

# Train the C,R -> A classifier, with ISNS weights
python train.py ./data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_ISNS_weighted_loss.pkl answer-extraction-2 3 3 --CRA 

# Train C -> R classifier
# python train.py ./data/tokenized_CR_data ./results/model_CR_5_epochs.pkl sentence-extraction 3 5