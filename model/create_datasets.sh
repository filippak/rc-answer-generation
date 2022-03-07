# Embeddings for C -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_training_data.pkl ./data/tokenized_train_data

# Embeddings for C -> A token classifier, test data
python dataset.py ../data-analysis/create_dataset/data/labeled_test_data.pkl ./data/tokenized_test_data

# Embeddings for C,A -> R token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/labeled_sentence_extraction_training_data.pkl ./data/tokenized_sentence_extraction_train_data --answers