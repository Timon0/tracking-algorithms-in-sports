# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import sys
import time

import torch

USE_TENSORBOARD = False
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.exists(opt.debug_dir):
            os.makedirs(opt.debug_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)


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
                logger.info(
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

    def save_checkpoint(self, path, metadata=None):
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
        wandb_params.update({'name': args.exp_id})
        wandb_params.update({'save_dir': args.save_dir})

        return cls(config=vars(args), **wandb_params)
