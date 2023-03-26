import logging
from functools import partial

import hydra
import wandb
import albumentations as A
from omegaconf import DictConfig, OmegaConf
from fastai.data.transforms import Normalize
from fastai.vision.augment import Resize, imagenet_stats
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, error_rate, top_k_accuracy, F1Score
from fastai.optimizer import Adam  # , LabelSmoothingCrossEntropy

# from fastai.callback.mixup import MixUp
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

    log.info(f"Length of DF: {len(df)} rows.")

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

    item_tfms = [
        Resize(cfg.data.img_presize),
        AlbumentationsTransform(get_train_aug(), get_valid_aug()),
    ]

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

    log.info(f"Steps in train_dl: {len(dls.train)}")
    log.info(f"Steps in val_dl: {len(dls.valid)}")

    log.info(f"Classes being trained on: {dls.vocab}")

    top_3_accuracy = partial(top_k_accuracy, k=3)
    learn = vision_learner(
        dls,
        cfg.train.arch,
        metrics=[error_rate, accuracy, top_3_accuracy, F1Score(average="macro")],
        cbs=[MixedPrecision()],
        wd=cfg.train.wd,
        opt_func=Adam,
        # loss_func=LabelSmoothingCrossEntropy(),
    )

    if cfg.load:
        log.info(f"Loading state dict and optimizer state from: models/{cfg.load}.pth")
        learn.load(f"train/{cfg.load}")

    if cfg.lr_find:
        log.info("Trying to find suitable learning rate")
        lr = learn.lr_find()
        log.info(f"Found suitable learning rate: {lr.valley:.2e}")
        return
    lr = cfg.train.lr

    # callbacks passed during each run like wandb
    cbs = [SaveModelCallback(monitor="f1_score")]

    # ugly coding, will look into refractoring this part
    if cfg.track:
        wandb.init(project=cfg.project, name=cfg.track_name)
        cbs.append(WandbCallback())

    log.info(f"Training model for {cfg.train.freeze_epochs} initial epochs")
    log.info(f"Unfreezing, and training for a further {cfg.train.epochs} epochs.")
    # learn.fine_tune(
    #     epochs=cfg.train.epochs,
    #     freeze_epochs=cfg.train.freeze_epochs,
    #     base_lr=lr,
    #     cbs=cbs,
    # )
    # learn.fit_one_cycle(cfg.train.epochs, lr, cbs=cbs)

    if cfg.track:
        wandb.finish(quiet=True)

    if cfg.save:
        log.info(
            f"Saving state dict and optimizer state to: models/train/{cfg.save}.pth"
        )
        learn.save(f"train/{cfg.save_model}")

    if cfg.export:
        log.info(f"Exporting model to: models/learners/{cfg.export}.pkl")
        learn.export(f"models/learners/{cfg.export}.pkl")


if __name__ == "__main__":
    wandb.login()
    main()
