from functools import partial

from PIL import ImageFile

import pandas as pd
from fastai.vision.augment import Resize
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, error_rate
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle
from fastcore.foundation import L

# from metaflow import Parameter, FlowSpec, step

from tsp_cls.utils.root import get_data_root
from tsp_cls.utils.data import get_image_path, get_genus, add_species
from tsp_cls.dataloader.dataloader import get_dls


ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    path = get_data_root()

    df = pd.read_csv(path / "SnakeCLEF2021_train_metadata_PROD.csv")
    df["species"] = df.apply(add_species, axis=1)

    # Taking a sample of the classes for the baseline
    to_consider = df.genus.value_counts()[:10]
    classes = L(to_consider.keys().tolist())

    sample_df = df.loc[df.genus.isin(classes)]
    print(f"Length of DF: {len(sample_df)}")

    item_tfms = [Resize(224)]

    dls = get_dls(
        df,
        get_x=partial(
            partial(get_image_path, data_path=path), data_path=get_data_root()
        ),
        get_y=get_genus,
        item_tfms=item_tfms,
        bs=64,
    )

    print(f"Steps in train_dl: {len(dls.train)}")

    learn = vision_learner(
        dls, "resnet18", metrics=[error_rate, accuracy], cbs=[MixedPrecision()]
    )

    learn.fit_one_cycle(1, 1e-3)


if __name__ == "__main__":
    main()
