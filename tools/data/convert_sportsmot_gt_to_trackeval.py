import os
import shutil

TRACKEVAL_DATA_ROOT = "TrackEval"
SPORTSMOT_DATA_ROOT = "datasets/SportsMOT"
MY_BENCHMARK_NAME = "SportsMOT"
NO_SUBDIR = True

for split in ["train", "val", "test"]:
    # ========default hierarchy========
    # folder hierarchy according to
    # https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data
    eval_gt_dir = os.path.join(os.getcwd(), TRACKEVAL_DATA_ROOT, "data", "gt", "mot_challenge")
    os.makedirs(eval_gt_dir, exist_ok=True)
    eval_seqmap_dir = os.path.join(os.getcwd(), TRACKEVAL_DATA_ROOT, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(eval_seqmap_dir, exist_ok=True)

    # create symlink for data
    source_split_dir = os.path.join(os.getcwd(), SPORTSMOT_DATA_ROOT, split)
    target_split_dir = os.path.join(eval_gt_dir, f"{MY_BENCHMARK_NAME}-{split}")
    os.symlink(source_split_dir, target_split_dir, target_is_directory=True)

    # copy split file
    split_txt_source_file = os.path.join(SPORTSMOT_DATA_ROOT, "splits_txt", f"{split}.txt")
    split_txt_target_file = os.path.join(eval_seqmap_dir, f"{MY_BENCHMARK_NAME}-{split}.txt")
    shutil.copy(split_txt_source_file,split_txt_target_file)
