# Load necessary libraries
import pandas as pd
import codecs
import argparse


def main(args):
    df = pd.read_json(codecs.open(args.data_path_1, 'r', 'utf-8'), orient='split')
    if not args.single_file:
        df_2 = pd.read_json(codecs.open(args.data_path_2, 'r', 'utf-8'), orient='split')

        # merge the two datasets to use: 
        dfs = [df, df_2]
        df = pd.concat(dfs)
        print(len(df))

    # save dataframe
    df.to_pickle(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset with labels')

    # command-line arguments
    parser.add_argument('data_path_1', type=str, 
        help='path to first json file', action='store')
    parser.add_argument('data_path_2', type=str, 
        help='path to second json file', action='store')
    parser.add_argument('output_path', type=str, 
        help='path to output file where the parsed data will be stored', action='store')
    parser.add_argument('--single', dest='single_file', action='store_true')


    args = parser.parse_args()
    main(args)
