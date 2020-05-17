""" Contains most components of the preprocessing pipeline. Cleans text,
tokenizes it, lower casing and identifying medical concepts.

Note that metamap server needs to be running in the background, for this code
to work.

@author: Nikhil Pattisapu, iREL, IIIT-H"""


import re
import string
import sys
from multiprocessing.pool import Pool
from nltk import word_tokenize
from nltk.corpus import words
from pymetamap import MetaMap
import numpy as np


BAR = 0.03
MIN_LENGTH = 6
MAX_LENGTH = 40

# pylint: disable=invalid-name, len-as-condition, too-many-locals

vocab = set(words.words())
mm_home = '/home/nikhil.pattisapu/tools/metamap2016/public_mm/bin/metamap16'
mm = MetaMap.get_instance(mm_home)
sem_types = ['antb', 'bhvr', 'bmod', 'blor', 'bdsu', 'bdsy', 'chem', 'clna',
             'cnce', 'clnd', 'dsyn', 'enty', 'evnt', 'fndg', 'food', 'ftcn',
             'hlca', 'hlco', 'idcn', 'inch', 'ocdi', 'ocac', 'bpoc', 'orch',
             'podg', 'phsu', 'phpr', 'lbpr', 'resa', 'resd', 'sbst', 'sosy',
             'tmco']
mm_threshold = 2


def remove_over_punctuated(sents):
    """Unused: Remove sentences which contain greater than BAR % of punctuation
    characters"""
    res = []
    for sent in sents:
        if len(sent) == 0:
            continue
        count = 0
        for char in sent.strip():
            if char in string.punctuation:
                count += 1
        if count / len(sent) < BAR:
            res.append(sent)
    return res


def ensure_proper_sent_len(sents):
    """Prune sentences which are either too short or are too long"""
    res = []
    for sent in sents:
        length = len(word_tokenize(sent.strip()))
        if length >= MIN_LENGTH and length <= MAX_LENGTH:
            res.append(sent)
    return res


def normalize_whitespace(sent):
    """Substitutes extra contiguous whitespaces with a single whitespace"""
    return ' '.join(sent.strip().split())


def remove_text_inside_brackets(text, brackets="()[]{}"):
    """Removes all content inside brackets"""
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for idx, brack in enumerate(brackets):
            if character == brack: # found bracket
                kind, is_close = divmod(idx, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)


def bio_asq_tokenization(sent):
    """Tokenize text based on BioASQ style. Some modifications are also done"""
    sent = sent.replace('%', ' percent')
    sent = sent.replace('&', ' and')
    sent = sent.replace("\u2013", " ")
    sent = sent.replace('"', '').replace('\\', '').strip().lower()
    sent = ' '.join(re.sub(r'[?;*!^_+():-\[\]{}]', ' ', sent).split())
    return sent


def get_metamap_op(sents):
    """Replace medical concept mentions with their corresponding UMLS Preferred
    Names"""
    res = []
    for sent in sents:
        sent = sent.strip()
        mm_sent = ''
        concepts, _error = mm.extract_concepts([sent])
        # print(concepts)
        concepts = [c for c in concepts if c[8].count('/') == 1]
        # concepts = [c for c in concepts if any([i in c[5] for i in sem_types])]
        fil_concepts = []
        for concept in concepts:
            try:
                score = float(concept[2])
                if score > mm_threshold:
                    fil_concepts.append(concept)
            except ValueError:
                pass

        # If an identified phrase is mapped to more than one concept, consider
        # the one with the highest score.
        unique_concepts = {}
        for concept in fil_concepts:
            if concept[8] not in unique_concepts:
                unique_concepts[concept[8]] = concept
            else:
                prev_score = unique_concepts[concept[8]][2]
                curr_score = concept[2]
                if curr_score > prev_score:
                    unique_concepts[concept[8]] = concept
        keys = list(unique_concepts.keys())
        sorted_idx = np.argsort([int(key.split('/')[0]) for key in keys])
        ordered_concepts = [unique_concepts[keys[idx]] for idx in sorted_idx]
        start_idx = 0
        for concept in ordered_concepts:
            parts = concept[8].split('/')
            end_idx = int(parts[0]) - 1
            mm_sent += sent[start_idx: end_idx]
            mm_sent += concept[3]
            start_idx = end_idx + int(parts[1])

        # If no concepts are found, copy everything!
        if start_idx != 0:
            mm_sent += sent[start_idx: len(sent)]
        else:
            mm_sent = sent
        res.append(mm_sent)
    return res


if __name__ == '__main__':
    ip_path, src_path, tgt_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sentences = [l.strip() for l in open(ip_path, 'r').readlines()]
    sentences = remove_over_punctuated(sentences)
    sentences = remove_text_inside_brackets(sentences)
    sentences = ensure_proper_sent_len(sentences)
    sentences = [normalize_whitespace(sent) for sent in sentences]
    sentences = [remove_text_inside_brackets(sent) for sent in sentences]
    sentences = [bio_asq_tokenization(sent) for sent in sentences]
    p = Pool(100)
    ops = p.map(get_metamap_op, sentences)
    sentences = [sent + "\n" for sent in sentences]
    ops = [sent + "\n" for sent in ops]
    with open(src_path, 'w') as src_file:
        src_file.writelines(sentences)
    with open(tgt_path, 'w') as tgt_file:
        tgt_file.writelines(ops)
