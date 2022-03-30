echo 'Training the C -> A token classifier model..'
python train.py ./data/CA/tokenized_CA_data ./results/CA/model_CA_3e_INS15.pkl answer-extraction 3 3

echo 'Training the CR -> A token classifier model..'
python train.py ./data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_INS15.pkl answer-extraction-2 3 3 --CRA 

echo 'Training the CA -> R sentence classifier model..'
python train_CAR_sent_class.py ./data/CAR_classification/data_subset/CAR_class_data ./results/CAR_classification/model_CAR_CLASS_3e_INS15_subset.pkl sentence-classification 2 3
