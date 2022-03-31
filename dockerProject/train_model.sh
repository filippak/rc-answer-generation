echo 'Training the C,A -> R sentence classifier on INS weights'
python3 train_CAR_sent_class.py data/CAR_classification/tokenized_CAR_class_data results/CAR_classification/model_CAR_CLASS_3e_INS.pkl sentence-classification 2 3
