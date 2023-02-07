# TODO: Update this file with updates from the test pipeline
from functools import partial

import pandas as pd
from fastai.vision.augment import Resize
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, error_rate
from fastai.callback.fp16 import MixedPrecision
from fastcore.foundation import L
from metaflow import Parameter, FlowSpec, step

from tsp_cls.utils.root import get_data_root
from tsp_cls.utils.data import get_image_path, get_genus, add_species
from tsp_cls.dataloader.dataloader import get_dls


class SmallBaselines(FlowSpec):
    """
    This flow trains a small baseline with only 10 classes.
    It forms a baseline to improve on.
    """

    classes = Parameter(
        "classes",
        help="""Number of classes to train on. 
        They are picked in descending order of the most common classes""",
        default=10,
    )
    arch = Parameter(
        "architecture", help="Architecture to train", default="convnext_small"
    )

    @step
    def start(self):
        "Setting up important variables"
        self.path = get_data_root()
        self.next(self.dataloading)

    @step
    def dataloading(self):
        "Loading our data."
        df = pd.read_csv(self.path / "SnakeCLEF2021_train_metadata_PROD.csv")
        df["species"] = df.apply(add_species, axis=1)

        # Taking a sample of the classes for the baseline
        to_consider = df.genus.value_counts()[:10]
        classes = L(to_consider.keys().tolist())

        self.sample_df = df.loc[df.genus.isin(classes)]
        print(f"Length of DF: {len(self.sample_df)}")

        item_tfms = [Resize(224)]

        self.dls = get_dls(
            df,
            get_x=partial(
                partial(get_image_path, data_path=self.path), data_path=get_data_root()
            ),
            get_y=get_genus,
            item_tfms=item_tfms,
            bs=64,
        )

        print(f"Steps in train_dl: {len(self.dls.train)}")

        self.next(self.train)

    @step
    def train(self):
        "Training the model step"
        print(self.arch)
        learn = vision_learner(
            self.dls, self.arch, metrics=[error_rate, accuracy], cbs=[MixedPrecision()]
        )

        learn.fit_one_cycle(10, 1e-3)
        self.next(self.end)

    @step
    def end(self):
        "End point"
        print("Finished")


if __name__ == "__main__":
    SmallBaselines()
