""" Evaluation Script: Computes BLEU, ROUGE, METEOR and SARI. """

import argparse
import os
import numpy as np
import ujson as json
from nlgeval import NLGEval
import sacrebleu
import rouge as rouge_score
from sari.SARI import SARIsent


class Metrics:
    """Class that is used to compute, store and write various NLG evaluation metrics to disk"""

    def __init__(self, source, target, hypothesis):
        super(Metrics, self).__init__()
        self.source = open(source, "r").readlines()
        self.target = open(target, "r").readlines()
        self.hypothesis = open(hypothesis, "r").readlines()
        self.metrics = {"name": os.path.basename(hypothesis)}

    def sacre_bleu(self):
        """ Computes BLEU using the sacrebleu library
            Link: https://github.com/mjpost/sacrebleu """
        bleu = {
            "BLEU": sacrebleu.corpus_bleu(
                self.hypothesis, [self.target], force=True
            ).score
        }
        self.metrics.update(bleu)

    def rouge(self):
        """ Computes ROUGE
            Link: https://github.com/Diego999/py-rouge """
        evaluator = rouge_score.Rouge(metrics=["rouge-n"], max_n=1)
        rouge = {
            "ROUGE": evaluator.get_scores(self.hypothesis, self.target)["rouge-1"]["f"]
            * 100
        }
        self.metrics.update(rouge)

    def meteor(self):
        """ Computes METEOR using the NLGEval library
            Link: https://github.com/Maluuba/nlg-eval"""
        metrics_to_omit = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"}
        nlgeval = NLGEval(
            no_skipthoughts=True, no_glove=True, metrics_to_omit=metrics_to_omit
        )
        self.metrics.update(nlgeval.compute_metrics([self.target], self.hypothesis))

    def sari(self):
        """ Method to compute SARI (System output Against References and against the Input sentence).
            Link: https://github.com/XingxingZhang/pysari """
        sari_score = []
        for source_line, target_line, hypothesis_line in zip(
            self.source, self.target, self.hypothesis
        ):
            sari_score.append(SARIsent(source_line, hypothesis_line, [target_line]))
        sari_score = {"SARI": np.array(sari_score).mean() * 100}
        self.metrics.update(sari_score)

    def evaluate(self):
        """ Method to run the evaluation """
        # self.bleu()
        self.sacre_bleu()
        self.sari()
        self.rouge()
        self.meteor()

    def write_to_disk(self, path):
        """ Method to write outputs to disk """
        with open(path, "a") as file:
            json.dump(self.metrics, file)
            file.write("\n")


def main():
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--hypothesis", type=str, required=True)
    args = parser.parse_args()

    metrics = Metrics(args.source, args.target, args.hypothesis)
    metrics.evaluate()
    print(metrics.metrics)


if __name__ == "__main__":
    main()
