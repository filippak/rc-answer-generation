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

# Embeddings for C -> R token classifier, training data (TEST!!!)
# python dataset.py ../data-analysis/create_dataset/data/labeled_sentence_extraction_training_data.pkl ./data/tokenized_CR_data