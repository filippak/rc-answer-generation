echo 'Run Quinductor on all test data'
python eval_qui_baseline.py ../../baseline_evaluation/Quinductor/data/qui_labeled_data.pkl ../../data-analysis/create_dataset/data/CA/labeled_CA_data_test.pkl

echo 'Run Quinductor on subset of test data'
python eval_qui_baseline.py ../../baseline_evaluation/Quinductor/data/qui_labeled_data.pkl ../../data-analysis/create_dataset/data/CA/labeled_CA_data_test.pkl --subset