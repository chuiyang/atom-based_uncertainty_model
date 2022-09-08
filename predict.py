"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
from pprint import pformat

if __name__ == '__main__':
    args = parse_predict_args()
    print(pformat(vars(args)))
    make_predictions(args)
