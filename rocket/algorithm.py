import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from DatasetConnector import ClassificationDatasetConnector
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.padder import PaddingTransformer


@dataclass
class CustomParameters:
    num_kernels: int = 2000
    random_state: int = 42  # seed for randomness


def train(args: ClassificationDatasetConnector):
    data = args.ts
    labels = args.labels
    padding = PaddingTransformer()
    data_transformed = padding.fit_transform(data)
    model = RocketClassifier(num_kernels=args.customParameters.num_kernels)
    model.fit(data_transformed, labels)
    joblib.dump(model, args.modelOutput)


def execute(args: ClassificationDatasetConnector):
    data = args.ts
    model = joblib.load(args.modelInput)
    padding = PaddingTransformer()
    data_transformed = padding.fit_transform(data)
    scores = model.predict(data_transformed)
    # scores = scores.astype(np.uint8)
    print(scores)
    scores = pd.factorize(scores, sort=True)[0].astype(np.uint8)
    # np.savetxt(args.dataOutput, scores, delimiter=",")
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: ClassificationDatasetConnector) -> None:
    seed = config.customParameters.random_state
    import random

    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = ClassificationDatasetConnector.from_sys_args(CustomParameters, sys.argv[1])
    set_random_state(args)
    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(
            f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'."
        )
