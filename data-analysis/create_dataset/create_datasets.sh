echo 'CREATING CA DATASET'

echo 'creating CA dataset for training data'
python create_CA_dataset.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ./data/labeled_CA_training_data.pkl

echo 'creating CA dataset for test data'
python create_CA_dataset.py ../data_frames/parsed_answer_data/df_test.pkl ./data/labeled_CA_test_data.pkl