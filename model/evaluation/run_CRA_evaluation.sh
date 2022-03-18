echo 'RUNNING EVALUATION ON C, R -> A MODEL'

echo 'evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/model_CRA_3e_weighted_loss.pkl ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS.pkl --CRA

echo 'STRICT evaluation of C, R -> A model, trained with INS weighted loss. '
python eval.py ../results/model_CRA_3e_weighted_loss.pkl ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS_strict.pkl --strict --CRA

echo 'evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_ISNS_weighted_loss.pkl ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS.pkl --CRA

echo 'STRICT evaluation of C, R -> A model, trained with ISNS weighted loss. '
python eval.py ../results/CRA/model_CRA_3e_ISNS_weighted_loss.pkl ../data/CRA/tokenized_CRA_data_eval_with_id.pkl ./data/CRA/tokenized_output_data_CRA_INS_strict.pkl --strict --CRA