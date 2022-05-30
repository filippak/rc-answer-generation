echo 'RUNNING EVALUATION ON C -> A MODEL'

echo 'Evaluation on selected model (INS partial) on test data'
python eval.py ../results/CA/fixed_seed/CA_3e_INS_fixed_seed ../data/CA/tokenized_CA_data_test_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS_test.pkl INS --token_eval

echo 'evaluation of C -> A model, trained with INS weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS.pkl INS --token_eval

echo 'STRICT evaluation of C -> A model, trained with INS weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS_strict.pkl INS --strict --token_eval


echo 'evaluation of C -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_ISNS_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_ISNS.pkl ISNS --token_eval

echo 'STRICT evaluation of C -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_ISNS_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_ISNS_strict.pkl ISNS --strict --token_eval


echo 'evaluation of C -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS15_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS15.pkl INS15 --token_eval

echo 'STRICT evaluation of C -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS15_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS15_strict.pkl INS15 --strict --token_eval


echo 'evaluation of C -> A model, trained with INS^(1.25) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS125_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS125.pkl INS125 --token_eval

echo 'STRICT evaluation of C -> A model, trained with INS^(1.25) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS125_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS125_strict.pkl INS125 --strict --token_eval


echo 'evaluation of C -> A model, trained with INS^(1.375) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS1375_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS1375.pkl INS1375 --token_eval

echo 'STRICT evaluation of C -> A model, trained with INS^(1.375) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS1375_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS1375_strict.pkl INS1375 --strict --token_eval


echo 'evaluation of C -> A model, trained with INS^(1.125) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS1125_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS1125.pkl INS1125 --token_eval

echo 'STRICT evaluation of C -> A model, trained with INS^(1.125) weighted loss. '
python eval.py ../results/CA/fixed_seed/CA_3e_INS1125_fixed_seed ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/fixed_seed/tokenized_output_data_CA_INS1125_strict.pkl INS1125 --strict --token_eval
