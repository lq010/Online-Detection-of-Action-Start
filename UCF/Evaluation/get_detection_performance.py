import argparse
import numpy as np

from Evaluation.eval_detection import UCFdetection

def main(ground_truth_filename, prediction_filename,
         tOffset_thresholds=np.linspace(0.2, 2.0, 10),
         verbose=True):

    anet_detection = UCFdetection(ground_truth_filename, prediction_filename,
                                    tOffset_thresholds=tOffset_thresholds,
                                   verbose=verbose)
    anet_detection.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--tOffset_thresholds', type=float, default=np.linspace(0.2, 2, 10),
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
