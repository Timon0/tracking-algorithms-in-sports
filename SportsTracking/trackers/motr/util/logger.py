import os

import torch


class WandbLogger(object):
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.
    By default, this information includes hyperparameters,
    system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    For more information, please refer to:
    https://docs.wandb.ai/guides/track
    https://docs.wandb.ai/guides/integrations/other/yolox
    """

    def __init__(self,
                 project=None,
                 name=None,
                 id=None,
                 entity=None,
                 save_dir=None,
                 config=None,
                 log_checkpoints=False,
                 **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            val_dataset (Dataset): validation dataset.
            num_eval_images (int): number of images from the validation set to log.
            log_checkpoints (bool): log checkpoints
            **kwargs: other kwargs.

        Usage:
            Any arguments for wandb.init can be provided on the command line using
            the prefix `wandb-`.
            Example
            ```
            python tools/train.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
            ```
            The val_dataset argument is not open to the command line.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
            )

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None

        self.log_checkpoints = (log_checkpoints == "True" or log_checkpoints == "true")
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config, allow_val_change=True)
        self.run.define_metric("train/epoch")
        self.run.define_metric("val/*", step_metric="train/epoch")
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")


    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                print(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def log_metrics(self, metrics, step=None):
        """
        Args:
            metrics (dict): metrics dict.
            step (int): step number.
        """

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        if step is not None:
            metrics.update({"train/step": step})
            self.run.log(metrics)
        else:
            self.run.log(metrics)

    def save_checkpoint(self, path, is_best=False, metadata=None):
        """
        Args:
            path (str): path.
            metadata (dict): metadata to save corresponding to the checkpoint.
        """

        if not self.log_checkpoints:
            return

        if "epoch" in metadata:
            epoch = metadata["epoch"]
        else:
            epoch = None

        filename = os.path.join(path)
        artifact = self.wandb.Artifact(
            name=f"run_{self.run.id}_model",
            type="model",
            metadata=metadata
        )
        artifact.add_file(filename, name="model_ckpt.pth.tar")

        aliases = ["latest"]

        if is_best:
            aliases.append("best")

        if epoch:
            aliases.append(f"epoch-{epoch}")

        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()

    @classmethod
    def initialize_wandb_logger(cls, args):
        wandb_params = dict()
        wandb_params.update({'project': args.wandb_project})
        wandb_params.update({'name': args.wandb_run_name})
        wandb_params.update({'save_dir': args.output_dir})

        return cls(config=vars(args), **wandb_params)