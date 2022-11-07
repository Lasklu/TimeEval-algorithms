from sktime.datasets import load_UCR_UEA_dataset, load_from_tsfile, load_from_tsfile_to_dataframe
import numpy as np
from pathlib import Path, WindowsPath, PosixPath
import json
import argparse


class ClassificationDatasetConnector(argparse.Namespace):
    @property
    def df(self):
        # return load_from_tsfile(Path(self.dataInput), return_data_type='numpy2d')
        return load_from_tsfile_to_dataframe(Path(self.dataInput))

    @property
    def ts(self) -> np.ndarray:
        return self.df[0]

    @property
    def labels(self) -> np.ndarray:
        return self.df[1]

    @staticmethod
    def from_sys_args(CustomParameters, args) -> 'ClassificationDatasetConnector':
        args: dict = json.loads(args)
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return ClassificationDatasetConnector(**args)
