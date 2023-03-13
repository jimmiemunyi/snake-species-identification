import logging
from functools import partial

import hydra
import wandb
import albumentations as A
from omegaconf import DictConfig, OmegaConf
from fastai.data.transforms import Normalize
from fastai.vision.augment import Resize, imagenet_stats
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, error_rate
from fastai.optimizer import Adam, LabelSmoothingCrossEntropy
from fastai.callback.mixup import MixUp
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.wandb import WandbCallback

from tsp_cls.utils.root import get_data_root
from tsp_cls.utils.data import (
    get_image_path,
    field_getter,
    read_dataframe,
    sample_dataframe,
)
from tsp_cls.dataloader.dataloader import get_dls
from tsp_cls.dataloader.augment import AlbumentationsTransform

log = logging.getLogger("pipelines.train")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    path = get_data_root()

    # TODO: fix repetition
    if cfg.data.classes == "full":
        df = read_dataframe(path, cfg.data.dataset)
    else:
        df = read_dataframe(path, cfg.data.dataset)
        df = sample_dataframe(df, cfg.train.get_y, cfg.data.classes)

    print(f"Length of DF: {len(df)}")

    # defining our augmentations
    # TODO: unpack the config variables at the top of this function
    img_size = cfg.data.img_size

    def get_train_aug():
        return A.Compose(
            [
                A.RandomResizedCrop(img_size, img_size),
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
            ]
        )

    def get_valid_aug():
        return A.Compose(
            [A.CenterCrop(img_size, img_size, p=1.0), A.Resize(img_size, img_size)],
            p=1.0,
        )

    item_tfms = [Resize(256), AlbumentationsTransform(get_train_aug(), get_valid_aug())]
    # item_tfms = [Resize(img_size)]

    batch_tfms = Normalize.from_stats(*imagenet_stats)
    dls = get_dls(
        df,
        get_x=partial(
            partial(get_image_path, data_path=path), data_path=get_data_root()
        ),
        get_y=partial(field_getter, field=cfg.train.get_y),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=cfg.dls.batch_size,
    )

    print(f"Steps in train_dl: {len(dls.train)}")
    print(f"Classes being trained on: {dls.vocab}")

    learn = vision_learner(
        dls,
        cfg.train.arch,
        metrics=[error_rate, accuracy],
        cbs=[MixedPrecision(), MixUp()],
        wd=cfg.train.wd,
        opt_func=Adam,
        loss_func=LabelSmoothingCrossEntropy(),
    )

    if cfg.lr_find:
        log.info("Trying to find suitable learning rate")
        lr = learn.lr_find()
        log.info(f"Found suitable learning rate: {lr.valley:.2e}")
        learn.unfreeze()
        lr = learn.lr_find()
        log.info(f"Found suitable learning rate after unfreezing: {lr.valley:.2e}")
        log.info("Better to pass in as a slice.")
        return
    lr = cfg.train.lr

    # callbacks passed during each run like wandb
    cbs = [SaveModelCallback()]

    # ugly coding, will look into refractoring this part
    if cfg.track:
        wandb.init(project=cfg.project, name=cfg.name)
        cbs.append(WandbCallback())

    log.info(f"Training model for {cfg.train.freeze_epochs} initial epochs")
    log.info(f"Unfreezing, and training for a further {cfg.train.epochs} epochs.")
    learn.fine_tune(
        epochs=cfg.train.epochs,
        freeze_epochs=cfg.train.freeze_epochs,
        base_lr=lr,
        cbs=cbs,
    )
    # learn.fit_one_cycle(cfg.train.epochs, lr, cbs=cbs)

    if cfg.track:
        wandb.finish(quiet=True)

    if cfg.save_model == "True":
        learn.export(fname=f"{cfg.cpt_name}")


if __name__ == "__main__":
    wandb.login()
    main()
