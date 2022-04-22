echo 'evaluation of C,A -> R model, trained with INS weighted loss. '
python eval_CAR_classification.py ../results/CAR_classification/model_CAR_CLASS_3e_INS ../data/CAR_classification/tokenized_CAR_class_data_eval_with_id.pkl INS

echo 'evaluation of C,A -> R model, trained with ISNS weighted loss. '
python eval_CAR_classification.py ../results/CAR_classification/model_CAR_CLASS_3e_ISNS ../data/CAR_classification/tokenized_CAR_class_data_eval_with_id.pkl ISNS

echo 'evaluation of C,A -> R model, trained with INS15 weighted loss. '
python eval_CAR_classification.py ../results/CAR_classification/model_CAR_CLASS_3e_INS15 ../data/CAR_classification/tokenized_CAR_class_data_eval_with_id.pkl INS15