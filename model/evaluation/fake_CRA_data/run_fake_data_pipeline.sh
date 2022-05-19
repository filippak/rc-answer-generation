

echo 'Creating fake CRA data'
python create_fake_CRA_data.py ../data/roundtrip/subset_10/test_data_CA_predictions.pkl ./fake_CRA_data.pkl

echo 'Tokenizing fake CRA data'
python ../../dataset.py ./fake_CRA_data.pkl ./tokenized_fake_CRA_data --CRA

echo 'Making CRA predictions on the fake data'
python ../CRA_roundtrip_predictions.py ../../results/CRA/fixed_seed/CRA_3e_ISNS_fixed_seed ./tokenized_fake_CRA_data_with_id.pkl ./fake_data_CRA_predictions.pkl


