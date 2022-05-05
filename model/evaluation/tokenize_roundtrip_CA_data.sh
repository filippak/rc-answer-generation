echo 'Label the test data based on the CA predictions'
# python label_using_CA_results.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/roundtrip/roundtrip_CAR_test_data.pkl

echo 'Select a subset of the context texts and label the data based on CA predictions'
python label_using_CA_results.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/roundtrip/roundtrip_CAR_test_subset_data.pkl --subset

echo 'Tokenize data based on output from CA model'
# python ../dataset_sent_classification_scheme_A.py ./data/roundtrip/roundtrip_CAR_test_data.pkl ./data/roundtrip/tokenized_CAR_test_data

echo 'Tokenize data based on subset output from CA model'
python ../dataset_sent_classification_scheme_A.py ./data/roundtrip/roundtrip_CAR_test_subset_data.pkl ./data/roundtrip/tokenized_CAR_test_subset_data

echo 'Make the CA-R predictions based on the data from the CA model and label for CRA model'
python make_CAR_roundtrip_predictions.py ../results/CAR_classification/token_type_ids/CAR_A_tti_balanced_3e_fixed_seed ./data/roundtrip/tokenized_CAR_test_data_with_id.pkl ./data/roundtrip/roundtrip_CRA_data.pkl

echo 'Tokenize the data based on the output from the CA-R model'
python ../dataset.py ./data/roundtrip/roundtrip_CRA_data.pkl ./data/roundtrip/tokenized_CRA_test_data --CRA

echo 'Make the CR-A predictions on the data from the CA-R model'
python CRA_roundtrip_predictions.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ./data/roundtrip/tokenized_CRA_test_data_with_id.pkl ./data/roundtrip/roundtrip_test_data_CRA_ISNS_strict.pkl

echo 'Make roundtrip check on the results produced by CA and CR-A models'
python extract_rc_answers_tokens.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/roundtrip/roundtrip_test_data_CRA_ISNS_strict.pkl


echo 'LABEL SUBSET'

echo 'Select a subset of the context texts and label the data based on CA predictions'
python label_using_CA_results.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/roundtrip/roundtrip_CAR_test_subset_data.pkl --subset

echo 'Tokenize data based on subset output from CA model'
python ../dataset_sent_classification_scheme_A.py ./data/roundtrip/roundtrip_CAR_test_subset_data.pkl ./data/roundtrip/tokenized_CAR_test_subset_data

echo 'Make the CA-R predictions based on the data subset from the CA model and label for CRA model'
python make_CAR_roundtrip_predictions.py ../results/CAR_classification/token_type_ids/CAR_A_tti_balanced_3e_fixed_seed ./data/roundtrip/tokenized_CAR_test_subset_data_with_id.pkl ./data/roundtrip/roundtrip_CRA_subset_data.pkl

echo 'Tokenize the data based on the output from the CA-R model'
python ../dataset.py ./data/roundtrip/roundtrip_CRA_subset_data.pkl ./data/roundtrip/tokenized_CRA_subset_test_data --CRA

echo 'Make the CR-A predictions on the data from the CA-R model'
python CRA_roundtrip_predictions.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ./data/roundtrip/tokenized_CRA_subset_test_data_with_id.pkl ./data/roundtrip/roundtrip_subset_test_data_CRA_ISNS_strict.pkl

echo 'Make roundtrip check on the results produced by CA and CR-A models'
python extract_rc_answers_tokens.py ./data/roundtrip/CA_data_subset.pkl ./data/roundtrip/roundtrip_subset_test_data_CRA_ISNS_strict.pkl