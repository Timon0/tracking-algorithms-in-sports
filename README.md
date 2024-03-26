# Tracking Algorithms in Sports

## Installation
```shell
git clone https://github.com/Timon0/tracking-algorithms-in-sports.git
cd tracking-algorithms-in-sports

conda create -n sports-tracking python=3.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch

conda activate sports-tracking

pip install -r requirements.txt
pip install cython pycocotools cython_bbox
python setup.py develop
```

## Data preparation

Download [SportsMOT](https://github.com/MCG-NJU/SportsMOT) and put them under `datasets` in the following structure:
```
datasets
   |——————SportsMOT
            └——————train
            └——————val
            └——————test

```

Then, you need to turn the datasets to COCO format and prepare data for evaluation with :

```shell
python tools/data/convert_sportsmot_to_coco.py
```

## Training

### YOLOX
The COCO pretrained YOLOX model can be downloaded from their [model zoo](https://github.com/Megvii-BaseDetection/YOLOX). After downloading the pretrained models, you can put them under `pretrained`.

Train pretrained yolox model on SportsMOT dataset 
```shell
python tools/train_yolox.py -f exps/example/mot/yolox_tiny_sportsmot.py -b 64 --fp16 -c pretrained/yolox_tiny.pth
```

**Weights & Biases for Logging**

To use W&B for logging, install wandb in your environment and log in to your W&B account using

```shell
pip install wandb
wandb login
```

Log in to your W&B account

To start logging metrics to W&B during training add the flag `--logger` to the previous command and use the prefix "wandb-" to specify arguments for initializing the wandb run.

```shell
python tools/train_yolox.py -f exps/example/mot/yolox_tiny_sportsmot.py -b 4 --fp16 -c pretrained/yolox_tiny.pth --logger wandb wandb-project <project name>
```

More WandbLogger arguments include

```shell
python tools/train_yolox.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_images <num-images> \
                wandb-log_checkpoints <bool>
```

More information available [here](https://docs.wandb.ai/guides/integrations/other/yolox).

## Tracking

### SORT

```shell
python tools/track/track_sort.py -f exps/example/mot/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

### DeepSORT

Before using DeepSort, download the ckpt.t7 file from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and save it to the folder `pretrained`.

```shell
python tools/track/track_deepsort.py -f exps/example/mot/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

### ByteTrack

```shell
python tools/track/track_bytetrack.py -f exps/example/mot/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

## Evaluation
Prepare data for evaluation with [TrackEval](https://github.com/JonathonLuiten/TrackEval).

```shell
python tools/data/convert_sportsmot_gt_to_trackeval.py
# Example to convert the tracker results of the SORT tracker on the validation split
python tools/data/convert_sportsmot_tracker_to_trackeval.py -s val -expn yolox_x_sportsmot -tracker sort
```

```shell
# Example to evaluate the tracking results of the SORT tracker of the validation split
python tools/evaluation/evaluate-sportsmot.py -s val -expn yolox_tiny_sportsmot -tracker bytetrack
```

## Visualisation
Use the command below to visualise tracking results. The generated files are stored in the `visualisation` folder. It's recommended to tweak the script's parameters beforehand.

```shell
python tools/visualisation/visualiser.py -s val -expn yolox_x_sportsmot -tracker sort -sequence v_00HRwkvvjtQ_c001
```


## Acknowledgement

The code is mainly based on [ByteTrack](https://github.com/ifzhang/ByteTrack), [YOLO](https://github.com/Megvii-BaseDetection/YOLOX) and [MixSort](https://github.com/MCG-NJU/MixSort).
