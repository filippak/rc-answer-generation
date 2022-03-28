echo 'evaluation of C,A -> R model, trained with INS weighted loss. '
python eval_CAR_classification.py ../results/CAR_classification/model_CAR_CLASS_3e_INS.pkl ../data/CAR_classification/tokenized_CAR_class_data_eval_with_id.pkl INS

python eval_CAR_classification.py ../results/CAR_classification/model_CAR_CLASS_3e_manual_weights_5000.pkl ../data/CAR_classification/tokenized_CAR_class_data_eval_with_id.pkl INS