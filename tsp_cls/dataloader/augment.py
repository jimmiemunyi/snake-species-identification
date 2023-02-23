from numpy import array
from fastai.vision.core import PILImage

from fastai.vision.augment import RandTransform
from fastcore.basics import store_attr


# got this example from: https://docs.fast.ai/tutorial.albumentations.html#using-different-transform-pipelines-and-the-datablock-api
class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=array(img))["image"]
        else:
            aug_img = self.valid_aug(image=array(img))["image"]
        return PILImage.create(aug_img)


if __name__ == "__main__":
    from functools import partial

    import albumentations as A
    from fastai.vision.augment import Resize, imagenet_stats
    from fastai.data.transforms import Normalize

    from tsp_cls.utils.root import get_data_root
    from tsp_cls.utils.data import (
        get_image_path,
        field_getter,
        read_dataframe,
        sample_dataframe,
    )
    from tsp_cls.dataloader.dataloader import get_dls

    path = get_data_root()
    df = read_dataframe(path, "SnakeCLEF2021_min-train_metadata_PROD.csv", nrows=100)
    df = sample_dataframe(df, "genus", 10)

    print(f"Length of DF: {len(df)}")

    # defining some transforms
    def get_train_aug():
        return A.Compose(
            [
                A.RandomResizedCrop(224, 224),
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
            ]
        )

    def get_valid_aug():
        return A.Compose([A.CenterCrop(224, 224, p=1.0), A.Resize(224, 224)], p=1.0)

    item_tfms = [Resize(256), AlbumentationsTransform(get_train_aug(), get_valid_aug())]
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    dls = get_dls(
        df,
        get_x=partial(
            partial(get_image_path, data_path=path), data_path=get_data_root()
        ),
        get_y=partial(field_getter, field="genus"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=10,
    )
    xb, yb = dls.one_batch()
    print(xb.shape, yb.shape)
