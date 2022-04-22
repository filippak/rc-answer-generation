echo 'RUNNING EVALUATION ON C -> A MODEL'

echo 'evaluation of C -> A model, trained with INS weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS.pkl INS

echo 'STRICT evaluation of C -> A model, trained with INS weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS_strict.pkl INS --strict


echo 'evaluation of C -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CA/model_CA_3e_ISNS ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS.pkl ISNS

echo 'STRICT evaluation of C -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CA/model_CA_3e_ISNS ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS_strict.pkl ISNS --strict


echo 'evaluation of C -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS15 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS15.pkl INS15

echo 'STRICT evaluation of C -> A model, trained with INS15 weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS15 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS15_strict.pkl INS15 --strict


echo 'evaluation of C -> A model, trained with INS^(1.25) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS125 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS125.pkl INS125

echo 'STRICT evaluation of C -> A model, trained with INS^(1.25) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS125 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS125_strict.pkl INS125 --strict


echo 'evaluation of C -> A model, trained with INS^(1.375) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS1375 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS1375.pkl INS1375

echo 'STRICT evaluation of C -> A model, trained with INS^(1.375) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS1375 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS1375_strict.pkl INS1375 --strict


echo 'evaluation of C -> A model, trained with INS^(1.125) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS1125 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS1125.pkl INS1125

echo 'STRICT evaluation of C -> A model, trained with INS^(1.125) weighted loss. '
python eval.py ../results/CA/model_CA_3e_INS1125 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS1125_strict.pkl INS1125 --strict
