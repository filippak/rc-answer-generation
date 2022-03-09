# Embeddings for C -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_training_data.pkl ./data/tokenized_data

# Embeddings for C -> A token classifier, test data
python dataset.py ../data-analysis/create_dataset/data/labeled_test_data.pkl ./data/tokenized_data_test.pkl --test

# Embeddings for C,A -> R token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_CAR_training_data.pkl ./data/tokenized_CAR_data --answers

# Embeddings for C -> R token classifier, training data (TEST!!!)
python dataset.py ../data-analysis/create_dataset/data/labeled_sentence_extraction_training_data.pkl ./data/tokenized_CR_data