# Tracking Algorithms in Sports

## Installation
```shell
git clone https://github.com/Timon0/tracking-algorithms-in-sports.git
cd tracking-algorithms-in-sports

conda create -n sports-tracking python=3.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch

conda activate sports-tracking

pip install -r requirements.txt
sh make.sh
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

Then, you need to turn the datasets to COCO format and prepare data for evaluation with:

```shell
python tools/data/convert_sportsmot_to_coco.py
```

FairMOT uses a file with all the images:

```shell
python tools/data/generate_labels_per_frame.py
```

## Training

### YOLOX
The pretrained YOLOX model can be downloaded from their [model zoo](https://github.com/Megvii-BaseDetection/YOLOX). After downloading the pretrained models, you can put them under `pretrained`.

Train pretrained yolox model on SportsMOT dataset 
```shell
python tools/train/train_yolox.py -f exps/yolox/yolox_tiny_sportsmot.py -b 32 --fp16 -c pretrained/yolox_tiny.pth
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
python tools/train/train_yolox.py -f exps/yolox/yolox_tiny_sportsmot.py -b 32 --fp16 -c pretrained/yolox_tiny.pth --logger wandb wandb-project <project name>
```

More WandbLogger arguments include

```shell
python tools/train/train_yolox.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_images <num-images> \
                wandb-log_checkpoints <bool>
```

More information available [here](https://docs.wandb.ai/guides/integrations/other/yolox).

### FairMOT

The pretrained FairMOT model can be downloaded from their [model zoo](https://github.com/ifzhang/FairMOT). After downloading the pretrained models, you can put them under `pretrained`.


```shell
python tools/train/train_fairmot.py mot --exp_id fairmot_sportsmot --gpus 0 --batch_size 32 --load_model 'pretrained/fairmot_dla34.pth' --num_epochs 60 --lr_step '50' --data_cfg 'SportsTracking/trackers/fairmot/cfg/sportsmot.json'
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
python tools/train/train_fairmot.py mot --exp_id fairmot_sportsmot --gpus 0 --batch_size 16 --load_model 'pretrained/fairmot_dla34.pth' --num_epochs 60 --lr_step '50' --data_cfg 'SportsTracking/trackers/fairmot/cfg/sportsmot.json' --logger wandb wandb-project <project name>
```

### MOTR

The pretrained FairMOT model can be downloaded from their [model zoo](https://github.com/megvii-research/MOTR). After downloading the pretrained models, you can put them under `pretrained`.

```shell
sh exps/motr/motr_sportsmot_train.sh
```

**Weights & Biases for Logging**

To use W&B for logging, install wandb in your environment and log in to your W&B account using

```shell
pip install wandb
wandb login
```

## Tracking

### SORT

```shell
python tools/track/track_sort.py -f exps/yolox/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

### DeepSORT

Before using DeepSort, download the ckpt.t7 file from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and save it to the folder `pretrained`.

```shell
python tools/track/track_deepsort.py -f exps/yolox/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

### ByteTrack

```shell
python tools/track/track_bytetrack.py -f exps/yolox/yolox_x_sportsmot.py -c pretrained/yolox_x_sports_train.pth.tar -b 1 --fp16 --fuse
```

### FairMot

```shell
python tools/track/track_fairmot.py mot --load_model pretrained/fairmot_sportsmot.pth --conf_thres 0.6
```

### MOTR

```shell
sh exps/motr/motr_sportsmot_track.sh
```

## Evaluation
Prepare data for evaluation with [TrackEval](https://github.com/JonathonLuiten/TrackEval).

```shell
python tools/data/convert_sportsmot_gt_to_trackeval.py
# Example to convert the tracker results of the SORT tracker on the validation split
python tools/data/convert_sportsmot_tracker_to_trackeval.py -s val -expn yolox_tiny_sportsmot -tracker bytetrack
```

```shell
# Example to evaluate the tracking results of the SORT tracker of the validation split
python tools/evaluation/evaluate-sportsmot.py -s val -sport football -expn yolox_tiny_sportsmot -tracker bytetrack
```

## Visualisation
Use the command below to visualise tracking results. The generated files are stored in the `visualisation` folder. It's recommended to tweak the script's parameters beforehand.

```shell
python tools/visualisation/visualiser.py -s val -expn yolox_x_sportsmot -tracker sort -sequence v_00HRwkvvjtQ_c001
```

## Demo
Use the command below to track and visualise players in a video. The generated files are stored in the `visualisation` folder. It's recommended to tweak the script's parameters beforehand.

```shell
python tools/demo/tracking.py video -f exps/yolox/yolox_tiny_sportsmot.py -c pretrained/yolox_tiny_sportsmot.pth.tar --fp16 --fuse --save_result --path ./videos/football.mp4
```

## Sports Field Registration for Football (POC)
For mapping detections on a 2D football field, [SCCvSD](https://github.com/lood339/SCCvSD) is utilized. A demo is available at `tools/demo/scc_v_sd.ipynb`. 

First, download the trained models from [here](https://docs.google.com/uc?export=download&id=1EaBmCzl4xnuebfoQnxU1xQgNmBy7mWi2) and save it to `pretrained/scc_v_sd`.


## Acknowledgement

The code is mainly based on [ByteTrack](https://github.com/ifzhang/ByteTrack), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [MixSort](https://github.com/MCG-NJU/MixSort), [FairMOT](https://github.com/ifzhang/FairMOT) and [MOTR](https://github.com/megvii-research/MOTR).
