
echo 'PARSE DATA FROM JSON FILES'

# Parse the cleaned data
echo 'parsing cleaned data..'
python parse_json_files.py ../../data/data_cleaned/training_clean.json ../../data/data_cleaned/qa_part3_clean.json ../../data/data_cleaned/dev_clean.json ../data_frames/parsed_json_data/df_train_cleaned.pkl

# Parse the original data
echo 'parsing original data..'
python parse_json_files.py ../../data/training.json ../../data/dataset-2022-03-09/qa_part3.json ../../data/dev.json ../data_frames/parsed_json_data/df_train_original.pkl

# Parse the test data
echo 'parsing test data..'
python parse_json_files.py ../../data/test.json null null ../data_frames/parsed_json_data/df_test.pkl --single


echo 'FIND ANSWERS IN TEXTS'

# find answers, cleaned data
echo 'finding answers in cleaned data..'
python parse_answers_and_context.py ../data_frames/parsed_json_data/df_train_cleaned.pkl ../data_frames/parsed_answer_data/df_train_cleaned.pkl 

# find answers, original data
echo 'finding answers in original data..'
python parse_answers_and_context.py ../data_frames/parsed_json_data/df_train_original.pkl ../data_frames/parsed_answer_data/df_train_original.pkl

# find answers, test data
echo 'finding answers in test data..'
python parse_answers_and_context.py ../data_frames/parsed_json_data/df_test.pkl ../data_frames/parsed_answer_data/df_test.pkl



echo 'DEPENDENCY PARSING'

# dependency parsing of cleaned data
echo 'dependency parsing of cleaned data..'
python dependency_parsing.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/dependency_parsed_data/df_train_cleaned.pkl 

# dependency parsing of original data
echo 'dependency parsing of original data..'
python dependency_parsing.py ../data_frames/parsed_answer_data/df_train_original.pkl ../data_frames/dependency_parsed_data/df_train_original.pkl

# dependency parsing of test data
echo 'dependency parsing of test data..'
python dependency_parsing.py ../data_frames/parsed_answer_data/df_test.pkl ../data_frames/dependency_parsed_data/df_test.pkl


echo 'CREATING LOCAL CORPUS'

echo 'creating corpus of the cleaned data..'
# python create_corpus.py ../data_frames/parsed_json_data/df_train_cleaned.pkl ../../context-corpus.txt


echo 'TF-IDF CALCULATION'

echo 'calculating and exporting tf-idf scores for training AND test data'
python tf-idf_calculation.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/parsed_answer_data/df_test.pkl ../data_frames/tf-idf/df_tf_idf

echo 'COLLECTING RELEVANT SENTENCES'

echo 'Finding relevant sentences for cleaned data..'
python relevant_sentences.py ../data_frames/parsed_answer_data/df_train_cleaned.pkl ../data_frames/dependency_parsed_data/df_train_cleaned.pkl ../data_frames/tf-idf/df_tf_idf_lemma_stop.pkl ../data_frames/relevant_sentences/cleaned_data/

echo 'Finding relevant sentences for original data..'
python relevant_sentences.py ../data_frames/parsed_answer_data/df_train_original.pkl ../data_frames/dependency_parsed_data/df_train_original.pkl ../data_frames/tf-idf/df_tf_idf_lemma_stop.pkl ../data_frames/relevant_sentences/original_data/

echo 'Finding relevant sentences for test data..'
python relevant_sentences.py ../data_frames/parsed_answer_data/df_test.pkl ../data_frames/dependency_parsed_data/df_test.pkl ../data_frames/tf-idf/df_tf_idf_lemma_stop_test.pkl ../data_frames/relevant_sentences/test_data/