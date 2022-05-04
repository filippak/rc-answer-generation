echo 'TEST evaluation of C,A -> R model, labeling scheme B'
python eval_CAR_classification.py ../results/CAR_classification/token_type_ids/CAR_A_tti_balanced_3e_fixed_seed ../data/CAR_classification/set_token_type/balanced/tokenized_CAR_class_data_test_with_id.pkl 'scheme B'


echo 'evaluation of C,A -> R model, labeling scheme A'
python eval_CAR_classification.py  ../results/CAR_classification/balanced/CAR_A_balanced_3e_fixed_seed ../data/CAR_classification/balanced/tokenized_CAR_class_data_eval_with_id.pkl 'scheme A'

echo 'evaluation of C,A -> R model, labeling scheme B'
python eval_CAR_classification.py ../results/CAR_classification/token_type_ids/CAR_A_tti_balanced_3e_fixed_seed ../data/CAR_classification/set_token_type/balanced/tokenized_CAR_class_data_eval_with_id.pkl 'scheme B'

echo 'evaluation of C,A -> R model, labeling scheme C'
python eval_CAR_classification.py ../results/CAR_classification/sent/balanced/CAR_B_balanced_3e_fixed_seed ../data/CAR_classification/sent/balanced/tokenized_CAR_class_data_eval_with_id.pkl 'scheme C'