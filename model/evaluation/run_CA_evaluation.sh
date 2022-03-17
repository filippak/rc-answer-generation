echo 'RUNNING EVALUATION ON C -> A MODEL'

echo 'evaluation of C -> A model, trained with INS weighted loss. '
# run evaluation of CA data, trained on INS weights in loss function
python eval.py ../results/model_CA_3e_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS.pkl

echo 'STRICT evaluation of C -> A model, trained with INS weighted loss. '
# run STRICT evaluation of CA data, trained on INS weights in loss function
python eval.py ../results/model_CA_3e_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_INS_strict.pkl --strict

echo 'evaluation of C -> A model, trained with ISNS weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/model_CA_3e_ISNS_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS.pkl

echo 'STRICT evaluation of C -> A model, trained with ISNS weighted loss. '
# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/model_CA_3e_ISNS_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval_with_id.pkl ./data/CA/tokenized_output_data_CA_ISNS_strict.pkl --strict