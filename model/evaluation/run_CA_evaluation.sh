echo 'RUNNING EVALUATION ON C -> A MODEL'

echo 'evaluation of C -> A model, trained with INS^(1.25) weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/CA/model_CA_3e_INS125 ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS125.pkl INS125

echo 'evaluation of C -> A model, trained with INS^(1.5) weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/CA/model_CA_3e_INS15.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS15.pkl INS15

echo 'STRICT evaluation of C -> A model, trained with INS^(1.5) weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/CA/model_CA_3e_INS15.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS15_srict.pkl INS15 --strict

echo 'evaluation of C -> A model, trained with INS weighted loss. '
# run evaluation of CA data, trained on INS weights in loss function
python eval.py ../results/model_CA_3e_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS.pkl INS

echo 'STRICT evaluation of C -> A model, trained with INS weighted loss. '
# run STRICT evaluation of CA data, trained on INS weights in loss function
python eval.py ../results/model_CA_3e_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS_strict.pkl INS --strict

echo 'evaluation of C -> A model, trained with ISNS weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/model_CA_3e_ISNS_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS.pkl ISNS

echo 'STRICT evaluation of C -> A model, trained with ISNS weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/model_CA_3e_ISNS_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS_strict.pkl ISNS --strict