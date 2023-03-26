import logging
from fastcore.xtras import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from fastai.learner import load_learner
from huggingface_hub import push_to_hub_fastai


models_path = Path("models/learners")
log = logging.getLogger("pipelines.deploy")


@hydra.main(version_base=None, config_path="../configs", config_name="deploy")
def deploy(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    learn = load_learner(f"{models_path}/{cfg.learner}.pkl")
    repo_id = f"{cfg.username}/{cfg.model}"
    log.info(msg=f"Pushing model to {repo_id}.")
    log.info(msg=f"Model name: {cfg.learner}.")
    push_to_hub_fastai(learner=learn, repo_id=repo_id)


if __name__ == "__main__":
    deploy()
