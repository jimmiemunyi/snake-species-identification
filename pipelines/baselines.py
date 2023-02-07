# TODO: Implement model saving logic
import logging
from functools import partial

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from fastai.vision.augment import Resize
from fastai.vision.learner import vision_learner
from fastai.metrics import accuracy, error_rate
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.wandb import WandbCallback

from tsp_cls.utils.root import get_data_root
from tsp_cls.utils.data import (
    get_image_path,
    field_getter,
    read_dataframe,
    sample_dataframe,
)
from tsp_cls.dataloader.dataloader import get_dls


log = logging.getLogger("pipelines.test")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    path = get_data_root()

    df = read_dataframe(path, "SnakeCLEF2021_min-train_metadata_PROD.csv")
    sampled_df = sample_dataframe(df, cfg.train.get_y, cfg.data.sample)

    print(f"Length of DF: {len(sampled_df)}")

    item_tfms = [Resize(cfg.data.img_size)]

    dls = get_dls(
        sampled_df,
        get_x=partial(
            partial(get_image_path, data_path=path), data_path=get_data_root()
        ),
        get_y=partial(field_getter, field=cfg.train.get_y),
        item_tfms=item_tfms,
        bs=cfg.dls.batch_size,
    )

    print(f"Steps in train_dl: {len(dls.train)}")
    print(f"Classes being trained on: {dls.vocab}")

    learn = vision_learner(
        dls,
        cfg.train.arch,
        metrics=[error_rate, accuracy],
        cbs=[MixedPrecision()],
    )
    lr = cfg.train.lr
    if lr == "None":
        log.info("Trying to find suitable learning rate")
        lr = learn.lr_find()
        log.info(f"Found suitable learning rate: {lr}")

    wandb.init(project=cfg.project, name=cfg.train.name)
    learn.fit_one_cycle(cfg.train.epochs, lr, cbs= WandbCallback())
    wandb.finish(quiet=True)


if __name__ == "__main__":
    wandb.login()
    main()
