echo 'RUNNING EVALUATION ON C, R -> A MODEL'

echo 'evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_ISNS ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_ISNS.pkl ISNS --CRA

echo 'STRICT evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_ISNS ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_ISNS_strict.pkl ISNS --strict --CRA

echo 'evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS.pkl INS --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS_strict.pkl INS --strict --CRA

echo 'evaluation of C, R -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS15 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS15.pkl INS15 --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS15 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS15_strict.pkl INS15 --strict --CRA

echo 'evaluation of C, R -> A model, trained with INS175 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS175 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS175.pkl INS175 --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS175 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS175 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS175_strict.pkl INS175 --strict --CRA

echo 'evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS1875 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS1875.pkl INS1875 --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS1875 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS1875_strict.pkl INS1875 --strict --CRA

echo 'evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS225 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS225.pkl INS225 --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS1875 weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_INS225 ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS225_strict.pkl INS225 --strict --CRA

echo 'evaluation of C, R -> A model, trained with ISNS weighted loss. BGN and END token labeled as -100 '
# python eval.py ../results/CRA/model_CRA_3e_ISNS_BGN_END_as_special.pkl ../data/CRA/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_ISNS_BGN_END_as_special.pkl ISNS --CRA

echo 'STRICT evaluation of C, R -> A model, trained with ISNS weighted loss. BGN and END token labeled as -100'
# python eval.py ../results/CRA/model_CRA_3e_ISNS_BGN_END_as_special.pkl ../data/CRA/tokenized_CRA_data_BGN_END_as_special_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_ISNS_BGN_END_as_special.pkl ISNS --strict --CRA