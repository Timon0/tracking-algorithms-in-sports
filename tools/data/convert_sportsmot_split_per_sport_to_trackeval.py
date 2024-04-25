import os
import shutil

TRACKEVAL_DATA_ROOT = "TrackEval/data/gt/mot_challenge/seqmaps"
SPORTSMOT_DATA_ROOT = "datasets/SportsMOT/splits_txt"
MY_BENCHMARK_NAME = "SportsMOT"

if __name__ == "__main__":
    split = 'val'
    sports = ['basketball', 'volleyball', 'football']

    for sport in sports:
        split_txt_source_file = os.path.join(SPORTSMOT_DATA_ROOT, f'{split}-{sport}.txt')
        split_txt_target_file = os.path.join(TRACKEVAL_DATA_ROOT, f"{MY_BENCHMARK_NAME}-{split}-{sport}.txt")
        shutil.copy(split_txt_source_file, split_txt_target_file)
