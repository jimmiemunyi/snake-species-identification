import logging
from functools import partial

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from fastai.data.transforms import Normalize
from fastai.vision.augment import Resize, RandomResizedCrop, imagenet_stats
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

    item_tfms = [Resize(460)]
    batch_tfms = [
        RandomResizedCrop(cfg.data.img_size, max_scale=0.75),
        Normalize.from_stats(*imagenet_stats),
    ]
    dls = get_dls(
        sampled_df,
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
        cbs=[MixedPrecision()],
    )
    lr = cfg.train.lr
    if lr == "None":
        log.info("Trying to find suitable learning rate")
        lr = learn.lr_find()
        log.info(f"Found suitable learning rate: {lr}")

    cbs = []

    # ugly coding, will look into refractoring this part
    if cfg.track:
        wandb.init(project=cfg.project, name=cfg.name)
        cbs.append(WandbCallback())

    learn.fit_one_cycle(cfg.train.epochs, lr, cbs=cbs)

    # if cfg.track:
    #     wandb.finish(quiet=True)

    if cfg.save_model == "True":
        learn.export(fname=f"{cfg.cpt_name}")


if __name__ == "__main__":
    wandb.login()
    main()
