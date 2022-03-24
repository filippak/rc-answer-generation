echo 'CREATING CA EMBEDDINGS'

echo 'embeddings training data..'
# Embeddings for C -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_train.pkl ./data/CA/tokenized_CA_data_train

echo 'embeddings evaluation data..'
# Embeddings for C -> A token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_eval.pkl ./data/CA/tokenized_CA_data_eval

echo 'embeddings test data..'
# Embeddings for C -> A token classifier, test data
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_test.pkl ./data/CA/tokenized_CA_data_test

echo 'CREATING CA -> R EMBEDDINGS'

echo 'embeddings training data..'
# Embeddings for C,A -> R token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/CAR/labeled_CAR_data_train.pkl ./data/CAR/tokenized_CAR_data_train --answers

echo 'embeddings evaluation data..'
# Embeddings for C,A -> R token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/CAR/labeled_CAR_data_eval.pkl ./data/CAR/tokenized_CAR_data_eval --answers

#TODO: Embeddings for C,A -> R token classifier, test data

echo 'CREATING CR -> A EMBEDDINGS'

echo 'embeddings training data..'
# Embeddings for C,R -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_train.pkl ./data/CRA/tokenized_CRA_data_train --CRA

echo 'embeddings evaluation data..'
# Embeddings for C,R -> A token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_eval.pkl ./data/CRA/tokenized_CRA_data_eval --CRA

echo 'embeddings training data, with BGN and END set to -100'
# Embeddings for C,R -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_train.pkl ./data/CRA/tokenized_CRA_data_BGN_END_as_special_train --CRA --CRA_tok_ignore

echo 'embeddings evaluation data, with BGN and END set to -100'
# Embeddings for C,R -> A token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_eval.pkl ./data/CRA/tokenized_CRA_data_BGN_END_as_special_eval --CRA --CRA_tok_ignore


echo 'CREATING CA -> R SENTENCE CLASSIFICATION EMBEDDINGS'

echo 'embeddings training data..'
# Embeddings for C,A -> R token classifier, training data
python dataset_sent_classification.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_train.pkl ./data/CAR_classification/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
# Embeddings for C,A -> R token classifier, evaluation data
python dataset_sent_classification.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_eval.pkl ./data/CAR_classification/tokenized_CAR_class_data_eval