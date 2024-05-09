import subprocess
import time

from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

if __name__ == "__main__":
    distributions = {
        "track_thresh": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "track_buffer": [10, 30, 50, 70, 90],
        "match_thresh": [0.8, 0.85, 0.9, 0.95],
    }

    sampler = ParameterSampler(distributions, n_iter=15, random_state=42)

    for iter, sample in enumerate(sampler):
        print(f"Start iteration {iter + 1}")
        start_time = time.time()
        track_thresh = str(sample['track_thresh'])
        track_buffer = str(sample['track_buffer'])
        match_thresh = str(sample['match_thresh'])
        subfolder = f"hyperparameter_{iter + 1}_track-thresh{track_thresh}_track-buffer{track_buffer}_match-thresh{match_thresh}"

        script = "tools/track/track_bytetrack.py"
        args = [
            # default
            "-f", "exps/yolox/yolox_tiny_sportsmot.py",
            "-c", "pretrained/yolox_tiny_sportsmot.pth.tar",
            "-b", "1",
            "--fp16",
            "--fuse",
            "--subfolder", subfolder,
            # sampling
            "--track_thresh", track_thresh,
            "--track_buffer", track_buffer,
            "--match_thresh", match_thresh
        ]

        subprocess.run(["python", script, *args])
        # python tools/data/convert_sportsmot_tracker_to_trackeval.py -s val -expn yolox_tiny_sportsmot -tracker bytetrack -subfolder hyperparameter_...
        subprocess.run(["python", "tools/data/convert_sportsmot_tracker_to_trackeval.py", "-s", "val", "-expn", "yolox_tiny_sportsmot", "-tracker", "bytetrack", "-subfolder", subfolder])
        # python tools/evaluation/evaluate-sportsmot.py -s val -expn yolox_tiny_sportsmot -tracker bytetrack -subfolder hyperparameter_...
        subprocess.run(["python", "tools/evaluation/evaluate-sportsmot.py", "-s", "val", "-expn", "yolox_tiny_sportsmot", "-tracker", "bytetrack", "-subfolder", subfolder])

        end_time = time.time()
        print(f"Iteration {iter + 1} finished in {end_time - start_time}s")

