# compare the answers extracted from C -> A model and C, R -> A model
echo 'Get roundtrip consistent answers from test data'
python extract_roundtrip_consistant_answers.py ./data/CA/fixed_seed/test_data_CA_INS.pkl ./data/CRA/fixed_seed/test_data_CRA_ISNS_strict.pkl 