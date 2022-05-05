echo 'LABEL SUBSET'

echo 'Select a subset of the context texts and label the data based on CA predictions'
python label_using_CA_results.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/roundtrip/subset_10/test_data --subset

echo 'Tokenize data based on subset output from CA model'
python ../dataset_sent_classification_scheme_A.py ./data/roundtrip/subset_10/test_data_CAR.pkl ./data/roundtrip/subset_10/tokenized_test_data_CAR

echo 'Make the CA-R predictions based on the data subset from the CA model and label for CRA model'
python make_CAR_roundtrip_predictions.py ../results/CAR_classification/token_type_ids/CAR_A_tti_balanced_3e_fixed_seed ./data/roundtrip/subset_10/tokenized_test_data_CAR_with_id.pkl ./data/roundtrip/subset_10/test_data_CRA.pkl

echo 'Tokenize the data based on the output from the CA-R model'
python ../dataset.py ./data/roundtrip/subset_10/test_data_CRA.pkl ./data/roundtrip/subset_10/tokenized_test_data_CRA --CRA

echo 'Make the CR-A predictions on the data from the CA-R model'
python CRA_roundtrip_predictions.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ./data/roundtrip/subset_10/tokenized_test_data_CRA_with_id.pkl ./data/roundtrip/subset_10/test_data_CRA_predictions.pkl

echo 'Make roundtrip check on the results produced by CA and CR-A models'
python extract_rc_answers_tokens.py ./data/roundtrip/subset_10/test_data_CA_predictions.pkl ./data/roundtrip/subset_10/test_data_CRA_predictions.pkl