import os
import shutil

TRACKEVAL_TRACKER_ROOT = os.path.join(os.getcwd(), "TrackEval/data/trackers/mot_challenge/SportsMOT-val/yolox_tiny_mix_det-track_results_sort")
TRACKEVAL_TRACKER_DATA_ROOT = os.path.join(TRACKEVAL_TRACKER_ROOT, "data")
SPORTSMOT_EXPERIMENT_DATA_ROOT = os.path.join(os.getcwd(), "outputs/yolox_tiny_mix_det/track_results_sort")
MY_BENCHMARK_NAME = "SportsMOT"
NO_SUBDIR = True
# ========default hierarchy========
# folder hierarchy according to
# https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data
os.makedirs(TRACKEVAL_TRACKER_ROOT, exist_ok=True)

# create symlink for data
os.symlink(SPORTSMOT_EXPERIMENT_DATA_ROOT, TRACKEVAL_TRACKER_DATA_ROOT, target_is_directory=True)