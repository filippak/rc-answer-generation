echo 'RUNNING EVALUATION ON C, R -> A MODEL'

# echo 'evaluation on TEST set'
# python eval.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ../data/CRA/tokenized_CRA_data_test_with_id.pkl ./data/CRA/fixed_seed/test_data_CRA_ISNS_strict.pkl ISNS --strict --CRA --token_eval


"""echo 'evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_ISNS.pkl ISNS --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_ISNS_strict.pkl ISNS --strict --CRA --token_eval


echo 'evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS.pkl INS --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS_strict.pkl INS --strict --CRA --token_eval

echo 'evaluation of C, R -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS15_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS15.pkl INS15 --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS15_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS15_strict.pkl INS15 --strict --CRA --token_eval

echo 'evaluation of C, R -> A model, trained with INS175 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS175_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS175.pkl INS175 --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS175 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS175_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS175_strict.pkl INS175 --strict --CRA --token_eval

echo 'evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS1875_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS1875.pkl INS1875 --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS1875_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS1875_strict.pkl INS1875 --strict --CRA --token_eval

echo 'evaluation of C, R -> A model, trained with INS225 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS225_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS225.pkl INS225 --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS225 weighted loss. '
python eval.py ../results/CRA/fixed_seed/CRA_3e_INS225_fixed_seed ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS225_strict.pkl INS225 --strict --CRA --token_eval"""

echo 'evaluation of C, R -> A model, trained with INS weighted loss. BGN and END token labeled as -100 '
python eval.py ../results/CRA/fixed_seed/BGN_END_CRA_3e_INS_fixed_seed ../data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS_BGN_END_as_special.pkl INS --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with INS weighted loss. BGN and END token labeled as -100'
python eval.py ../results/CRA/fixed_seed/BGN_END_CRA_3e_INS_fixed_seed ../data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_INS_BGN_END_as_special.pkl INS --strict --CRA --token_eval

echo 'evaluation of C, R -> A model, trained with ISNS weighted loss. BGN and END token labeled as -100 '
python eval.py ../results/CRA/fixed_seed/BGN_END_CRA_3e_ISNS_fixed_seed ../data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_ISNS_BGN_END_as_special.pkl ISNS --CRA --token_eval

echo 'STRICT evaluation of C, R -> A model, trained with ISNS weighted loss. BGN and END token labeled as -100'
python eval.py ../results/CRA/fixed_seed/BGN_END_CRA_3e_ISNS_fixed_seed ../data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/fixed_seed/tokenized_output_data_CRA_ISNS_BGN_END_as_special.pkl ISNS --strict --CRA --token_eval