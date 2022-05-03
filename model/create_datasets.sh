echo 'CREATING CA EMBEDDINGS'

echo 'embeddings training data..'
# Embeddings for C -> A token classifier, training data
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_train.pkl ./data/CA/tokenized_CA_data_train

echo 'embeddings evaluation data..'
# Embeddings for C -> A token classifier, evaluation data
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_eval.pkl ./data/CA/tokenized_CA_data_eval

echo 'embeddings test data..'
python dataset.py ../data-analysis/create_dataset/data/CA/labeled_CA_data_test.pkl ./data/CA/tokenized_CA_data_test


echo 'CREATING CR -> A EMBEDDINGS'

echo 'embeddings training data..'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_train.pkl ./data/CRA/tokenized_CRA_data_train --CRA

echo 'embeddings evaluation data..'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_eval.pkl ./data/CRA/tokenized_CRA_data_eval --CRA

echo 'embeddings test data..'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_test.pkl ./data/CRA/tokenized_CRA_data_test --CRA

echo 'embeddings training data, with BGN and END set to -100'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_train.pkl ./data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_train --CRA --CRA_tok_ignore

echo 'embeddings evaluation data, with BGN and END set to -100'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_eval.pkl ./data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_eval --CRA --CRA_tok_ignore

echo 'embeddings test data, with BGN and END set to -100'
python dataset.py ../data-analysis/create_dataset/data/CRA/labeled_CRA_data_test.pkl ./data/CRA/BGN_END/tokenized_CRA_data_BGN_END_as_special_test --CRA --CRA_tok_ignore


echo 'CREATING CA -> R SENTENCE CLASSIFICATION EMBEDDINGS, scheme A'

echo 'embeddings training data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_train.pkl ./data/CAR_classification/sent/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_eval.pkl ./data/CAR_classification/sent/tokenized_CAR_class_data_eval

echo 'embeddings test data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_test.pkl ./data/CAR_classification/sent/tokenized_CAR_class_data_test

echo 'ELECTRA embeddings training data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_train.pkl ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_train --electra

echo 'ELECTRA embeddings evaluation data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_eval.pkl ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_eval --electra

echo 'ELECTRA embeddings test data..'
python dataset_sent_classification_scheme_B.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_test.pkl ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_test --electra

echo 'CREATING CA -> R SENTENCE CLASSIFICATION EMBEDDINGS, scheme C'

echo 'embeddings training data..'
python dataset_sent_classification_scheme_A.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_train.pkl ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
python dataset_sent_classification_scheme_A.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_eval.pkl ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_eval

echo 'embeddings test data..'
python dataset_sent_classification_scheme_A.py ../data-analysis/create_dataset/data/CAR_classification/labeled_CAR_data_test.pkl ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_test
