echo 'BALANCING CA -> R SENTENCE CLASSIFICATION EMBEDDINGS DATA'

""" echo 'Balancing data where sentences are marked.. (scheme B)'

echo 'balancing training data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/tokenized_CAR_class_data_train ./data/CAR_classification/sent/balanced/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/tokenized_CAR_class_data_eval ./data/CAR_classification/sent/balanced/tokenized_CAR_class_data_eval

echo 'embeddings test data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/tokenized_CAR_class_data_test ./data/CAR_classification/sent/balanced/tokenized_CAR_class_data_test

echo 'ELECTRA embeddings training data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_train ./data/CAR_classification/sent/balanced/electra_tokenized_CAR_class_data_train

echo 'ELECTRA embeddings evaluation data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_eval ./data/CAR_classification/sent/balanced/electra_tokenized_CAR_class_data_eval

echo 'ELECTRA embeddings test data..'
# python make_balanced_sent_class.py ./data/CAR_classification/sent/electra_tokenized_CAR_class_data_test ./data/CAR_classification/sent/balanced/electra_tokenized_CAR_class_data_test


echo 'Balancing data where only BGN and END are marked..'

echo 'balancing training data..'
python make_balanced_sent_class.py ./data/CAR_classification/tokenized_CAR_class_data_train ./data/CAR_classification/balanced/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
python make_balanced_sent_class.py ./data/CAR_classification/tokenized_CAR_class_data_eval ./data/CAR_classification/balanced/tokenized_CAR_class_data_eval

echo 'embeddings test data..'
python make_balanced_sent_class.py ./data/CAR_classification/tokenized_CAR_class_data_test ./data/CAR_classification/balanced/tokenized_CAR_class_data_test

echo 'ELECTRA embeddings training data..'
python make_balanced_sent_class.py ./data/CAR_classification/electra_tokenized_CAR_class_data_train ./data/CAR_classification/balanced/electra_tokenized_CAR_class_data_train

echo 'ELECTRA embeddings evaluation data..'
python make_balanced_sent_class.py ./data/CAR_classification/electra_tokenized_CAR_class_data_eval ./data/CAR_classification/balanced/electra_tokenized_CAR_class_data_eval

echo 'ELECTRA embeddings test data..'
python make_balanced_sent_class.py ./data/CAR_classification/electra_tokenized_CAR_class_data_test ./data/CAR_classification/balanced/electra_tokenized_CAR_class_data_test """

echo 'Balancing data where only BGN and END are marked with token_type_ids (scheme A)'

echo 'balancing training data..'
python make_balanced_sent_class.py ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_train ./data/CAR_classification/set_token_type/balanced/tokenized_CAR_class_data_train

echo 'embeddings evaluation data..'
python make_balanced_sent_class.py ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_eval ./data/CAR_classification/set_token_type/balanced/tokenized_CAR_class_data_eval

echo 'embeddings test data..'
python make_balanced_sent_class.py ./data/CAR_classification/set_token_type/tokenized_CAR_class_data_test ./data/CAR_classification/set_token_type/balanced/tokenized_CAR_class_data_test