from transformers import pipeline, AutoTokenizer
import argparse
import pickle

def load_data(path):
    tokenizer = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
    contexts = []
    # The context texts are now the decoded versions of the text that was tokenized by the other bert model...
    for item in data:
        dec = tokenizer.decode(item["input_ids"][1:-1])
        tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        # print(dec)
        point = {'text': dec, 'original_input_ids': item["input_ids"], 'original_tokens': tokens, 'labels': item["labels"][1:-1]}
        contexts.append(point)
    return contexts

def main(args):
    contexts = load_data(args.data_path)
    # from: https://huggingface.co/KB/bert-base-swedish-cased-ner
    nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
    
    all_named_enteties = []
    for context in contexts:
        NE_in_context = []
        # print(context)
        l = []
        last_token_pos = None
        for token in nlp(context['text']):
            if token['word'].startswith('##') and len(l) > 0:
                l[-1]['word'] += token['word'][2:]
            elif last_token_pos and token['start'] - last_token_pos == 1: # if words are consequtive, add to same word
                l[-1]['word'] += ' ' + token['word']
            else:
                l += [token]

            last_token_pos = token['end']

        for ne in l:
            print('word: {}, entity: {}'.format(ne['word'], ne['entity']))
            NE_in_context.append(ne)
        all_named_enteties.append(NE_in_context)
        
        # TODO: evaluate against actual labels!


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune bert model for token classification')

    # command-line arguments
    parser.add_argument('data_path', type=str, 
        help='path to array of evaluation data', action='store')

    args = parser.parse_args()
    main(args)