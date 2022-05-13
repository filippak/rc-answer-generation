import argparse
import pickle
import pandas as pd
import stanza
import numpy as np
import matplotlib.pyplot as plt
import udon2
from udon2.visual import render_dep_tree


stanza.download('en', processors='tokenize,pos,lemma,depparse')
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
    

def main(args):
    doc = nlp('Tim plays basketball with friends and family every Tuesday')

    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

    # testing udon2 functionality
    roots = udon2.Importer.from_stanza(nlp("Tim plays basketball with friends and family every Tuesday").to_dict())
    print('roots: ', roots)
    root = roots[0]
    print('root:', root)
    node = root.select_by("id", "0")[0] # get the root node?
    render_dep_tree(root, "tree.svg") # node is an instance of udon2.Node
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Quinductor and extract suggested answer phrases')

    # command-line arguments

    args = parser.parse_args()
    main(args)

