echo 'CREATING CA EMBEDDINGS'

# Embeddings for C -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_CA_training_data.pkl ./data/tokenized_data

# Embeddings for C -> A token classifier, test data
python dataset.py ../data-analysis/create_dataset/data/labeled_CA_test_data.pkl ./data/tokenized_data_test.pkl --test

echo 'CREATING CA -> R EMBEDDINGS'
# Embeddings for C,A -> R token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_CAR_data_train.pkl ./data/tokenized_CAR_data_train.pkl --answers

# Embeddings for C,A -> R token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/labeled_CAR_data_eval.pkl ./data/tokenized_CAR_data_eval.pkl --answers

echo 'CREATING CR -> A EMBEDDINGS'
# Embeddings for C,R -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_CRA_data_train.pkl ./data/tokenized_CRA_data_train.pkl --CRA

# Embeddings for C,R -> A token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/labeled_CRA_data_eval.pkl ./data/tokenized_CRA_data_eval.pkl --CRA

# Embeddings for C -> R token classifier, training data (TEST!!!)
# python dataset.py ../data-analysis/create_dataset/data/labeled_sentence_extraction_training_data.pkl ./data/tokenized_CR_data