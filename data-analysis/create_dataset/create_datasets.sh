echo 'CREATING CA DATASET'

echo 'creating CA dataset for training data'
python create_CA_dataset.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ./data/labeled_CA_training_data.pkl

echo 'creating CA dataset for test data'
python create_CA_dataset.py ../data_frames/parsed_answer_data/df_test.pkl ./data/labeled_CA_test_data.pkl

echo 'CREATING CA->R DATASET'

echo 'creating CA->R dataset for training data'
python create_CAR_dataset.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/relevant_sentences/cleaned_data/lemma_stop.pkl ./data/labeled_CAR_data

# TODO: load the CA -> R dataset

echo 'CREATING CR->A DATASET'

echo 'creating CR->A dataset for training data'
python create_CRA_dataset.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/relevant_sentences/cleaned_data/lemma_stop.pkl ./data/labeled_CRA_data