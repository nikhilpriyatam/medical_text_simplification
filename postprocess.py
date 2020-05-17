"""Postprocessig script for models which use subword level tokenization
@author: Nikhil Pattisapu, iREL, IIIT-H"""

import sys

# pylint: disable=invalid-name, redefined-outer-name

def convert_tok_str_to_sent(tok_str):
    """Returns the sentence given the subword tokens"""
    res = ''
    tokens = tok_str.split(' ')
    for token in tokens:
        if '#' in token:
            token = token.replace('#', '')
            res += token
        else:
            res = res + ' ' + token
    return res.strip()


def clean_file(lines):
    """Returns the lines which contains only hypothesis"""
    lines = [line for line in lines if line.startswith('H-')]
    lines = [line.split('\t')[2] for line in lines]
    lines = [convert_tok_str_to_sent(line) + "\n" for line in lines]
    return lines


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Correct usage: python postprocess.py <input_file_path>"
              " <output_file_path>")
    ip_file = sys.argv[1]
    op_file = sys.argv[2]
    lines = open(ip_file, 'r').readlines()
    lines = clean_file(lines)
    with open(op_file, 'w') as op:
        op.writelines(lines)
