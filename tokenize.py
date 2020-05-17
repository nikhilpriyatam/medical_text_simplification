"""Tokenize text using the desired model.

Usage: python <inp_filepath> <vocab_filepath> <op_filepath>

* The format of input filepath is one sentence per line
* The format of vocab filepath is one token per line
* The format od output filepath is one tokenized sentence per line.

@author: Nikhil Pattisapu, iREL, IIIT-H"""


import sys
from multiprocessing import Pool
from functools import partial
from transformers import BertTokenizer


N_THREADS = 20 # Default number of threads


def tokenize(sent, bert_tok):
    """Return tokenized sentence"""
    return ' '.join(bert_tok.tokenize(sent))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python tokenize.py <in_file> <vocab_file> <op_file>")
    else:
        # pylint: disable=invalid-name
        pool = Pool(N_THREADS)
        model = BertTokenizer(sys.argv[2])
        sents = [s.strip().lower() for s in open(sys.argv[1], 'r').readlines()]
        p_tokenize = partial(tokenize, bert_tok=model)
        tok_sents = pool.map(p_tokenize, sents)
        tok_sents = [sent + "\n" for sent in tok_sents]
        with open(sys.argv[3], 'w') as tok_file:
            tok_file.writelines(tok_sents)
