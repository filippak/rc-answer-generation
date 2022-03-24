echo 'CREATING TRAIN AND EVALUATION SETS'

python divide_train_eval_dataset.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/relevant_sentences/cleaned_data/lemma_stop.pkl ./data/labeled_data

echo 'CREATING CA DATASET'

echo 'creating CA dataset for training data'
python create_CA_dataset.py ./data/labeled_data_train.pkl ./data/CA/labeled_CA_data_train.pkl

echo 'creating CA dataset for evaluation data'
python create_CA_dataset.py ./data/labeled_data_eval.pkl ./data/CA/labeled_CA_data_eval.pkl

echo 'creating CA dataset for test data'
python create_CA_dataset.py ../data_frames/parsed_answer_data/df_test.pkl ./data/CA/labeled_CA_data_test.pkl

echo 'CREATING CA->R DATASET'

echo 'creating CA->R dataset for training data'
python create_CAR_dataset.py ./data/labeled_data_train.pkl ./data/CAR/labeled_CAR_data_train.pkl

echo 'creating CA->R dataset for evaluation data'
python create_CAR_dataset.py ./data/labeled_data_eval.pkl ./data/CAR/labeled_CAR_data_eval.pkl

echo 'CREATING CR->A DATASET'

echo 'creating CR->A dataset for training data'
python create_CRA_dataset.py ./data/labeled_data_train.pkl ./data/CRA/labeled_CRA_data_train.pkl

echo 'creating CR->A dataset for evaluation '
python create_CRA_dataset.py ./data/labeled_data_eval.pkl ./data/CRA/labeled_CRA_data_eval.pkl


echo 'CREATING CA->R SENTENCE CLASSIFICATION DATASET'

echo 'creating CA->R sentence classification dataset for training data'
python create_CAR_sent_classification_dataset.py ./data/labeled_data_train.pkl ./data/CAR_classification/labeled_CAR_data_train.pkl

echo 'creating CA->R sentence classification dataset for evaluation data'
python create_CAR_sent_classification_dataset.py ./data/labeled_data_eval.pkl ./data/CAR_classification/labeled_CAR_data_eval.pkl