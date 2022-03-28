# Train the C -> A classifier, INS weights
python train.py ./data/CA/tokenized_CA_data ./results/model_CA_3e_INS_weighted_loss.pkl answer-extraction 3 3

# Train the C -> A classifier, ISNS weights
python train.py ./data/CA/tokenized_CA_data ./results/model_CA_3e_ISNS_weighted_loss.pkl answer-extraction 3 3

# Train the C,A -> R classifier
python train.py ./data/CAR/tokenized_CAR_data ./results/model_CAR_3e_INS_weighted_loss.pkl sentence-extraction 3 3 --CAR

# Train the C,A -> R classifier, ISNS weights
python train.py ./data/CAR/tokenized_CAR_data ./results/CAR/model_CAR_3e_ISNS_weighted_loss.pkl sentence-extraction 3 3 --CAR

# Train the C,R -> A classifier
python train.py ./data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_INS_weighted_loss.pkl answer-extraction-2 3 3 --CRA 

# Train the C,R -> A classifier, with ISNS weights
python train.py ./data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_ISNS_weighted_loss.pkl answer-extraction-2 3 3 --CRA 

# Train the C,R -> A classifier, with ISNS weights, with labels of BGN and END tokens set to -100
python train.py ./data/CRA/tokenized_CRA_data_BGN_END_as_special ./results/CRA/model_CRA_3e_ISNS_BGN_END_as_special.pkl answer-extraction-2 3 3 --CRA 

echo 'Training the C,A -> R sentence classifier..'
python train_CAR_sent_class.py ./data/CAR_classification/tokenized_CAR_class_data ./results/CAR_classification/model_CAR_CLASS_3e_ISNS_5000.pkl sentence-classification 2 3
