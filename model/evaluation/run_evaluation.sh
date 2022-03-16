# run evaluation of CA data, trained on INS weights in loss function
python eval.py ../results/model_CA_3e_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval.pkl

# run evaluation of CA data, trained on ISNS weights in loss function
python eval.py ../results/model_CA_3e_ISNS_weighted_loss.pkl ../data/CA/tokenized_CA_data_eval.pkl

# run evaluation of CAR data, trained on INS weights in loss function
python eval.py ../results/model_CAR_3e_weighted_loss.pkl ../data/CAR/tokenized_CAR_data_eval.pkl

# run evaluation of CRA data, trained on INS weights in loss function
python eval.py ../results/model_CRA_3e_weighted_loss.pkl ../data/CRA/tokenized_CRA_data_eval.pkl