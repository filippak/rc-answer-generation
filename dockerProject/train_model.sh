echo 'Training the C -> A token classifier on INS125 weights'
python train.py data/CA/tokenized_CA_data ./results/CA/model_CA_3e_INS125 answer-extraction 3 3

echo 'Training the C,R -> A token classifier on INS175 weights'
# python train.py data/CRA/tokenized_CRA_data ./results/CRA/model_CRA_3e_INS175.pkl answer-extraction-2 3 3 --CRA 

echo 'Training the C,A -> R sentence classifier on INS15 weights'
# python3 train_CAR_sent_class.py data/CAR_classification/tokenized_CAR_class_data results/CAR_classification/model_CAR_CLASS_3e_INS.pkl sentence-classification 2 3
