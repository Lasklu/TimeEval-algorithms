from dataclasses import dataclass
import numpy as np
import joblib
from sktime.classification.kernel_based import RocketClassifier
from DatasetConnector import ClassificationDatasetConnector
import sys


@dataclass
class CustomParameters:
    num_kernels: int = 2000
    random_state: int = 42  # seed for randomness


def train(args: ClassificationDatasetConnector):
    data = args.ts
    labels = args.labels
    model = RocketClassifier(num_kernels=args.customParameters.num_kernels)
    model.fit(data, labels)
    joblib.dump(model, args.modelOutput)


def execute(args: ClassificationDatasetConnector):
    data = args.ts
    model = joblib.load(args.modelInput)
    scores = model.predict(data)
    scores = scores.astype(np.uint8)
    print(scores)
    #np.savetxt(args.dataOutput, scores, delimiter=",")
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: ClassificationDatasetConnector) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = ClassificationDatasetConnector.from_sys_args(
        CustomParameters, sys.argv[1])
    set_random_state(args)
    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(
            f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
