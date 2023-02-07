from fastai.vision.data import ImageBlock, CategoryBlock
from fastai.data.block import DataBlock
from fastai.data.transforms import RandomSplitter


def get_dls(
    df, get_x, get_y, item_tfms, batch_tfms=None, valid_pct=0.2, bs=32
) -> DataBlock:
    """Creates and returns the dataloader."""
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=valid_pct),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    return dblock.dataloaders(df, bs=bs)


if __name__ == "__main__":

    from functools import partial

    from fastai.vision.augment import Resize

    from tsp_cls.utils.root import get_data_root
    from tsp_cls.utils.data import (
        get_image_path,
        get_genus,
        read_dataframe,
        sample_dataframe,
    )

    path = get_data_root()
    df = read_dataframe(path, "SnakeCLEF2021_min-train_metadata_PROD.csv", nrows=500)
    sampled_df = sample_dataframe(df, "genus", 3)
    print(f"Length of DF: {len(df)}")

    item_tfms = [Resize(224)]

    dls = get_dls(
        sampled_df,
        get_x=partial(get_image_path, data_path=get_data_root()),
        get_y=get_genus,
        item_tfms=item_tfms,
        bs=2,
    )

    print(f"Steps in train_dl: {len(dls.train)}")
    print(f"Classes being trained on:{dls.vocab}") # should be equal to the sample size
