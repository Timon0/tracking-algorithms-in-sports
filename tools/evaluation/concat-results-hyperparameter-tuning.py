import os

import pandas as pd
import re


METRICS = ['HOTA', 'MOTA', 'IDF1', 'IDSW']

def get_hyperparameter_trials(base_dir):
    all_items = os.listdir(base_dir)
    hyper_dirs = [item for item in all_items if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('hyperparameter')]

    hyperparameter_trial_list = []
    for directory in hyper_dirs:
        id = int(re.search(r"hyperparameter_(\d+)_", directory).group(1))
        hyperparameter_string = re.sub(r"hyperparameter_\d+_", "", directory)
        matches = re.findall(r'([a-z-]+)([\d.]+)', hyperparameter_string)

        hyperparameter_trial = {key: float(value) if '.' in value else int(value) for key, value in matches}
        hyperparameter_trial['id'] = id
        hyperparameter_trial['folder'] = directory
        hyperparameter_trial_list.append(hyperparameter_trial)

    return hyperparameter_trial_list

def getResults(base_dir, hyperparameter_trials):
    results = []
    for hyperparameter_trial in hyperparameter_trials:
        tracker_results_path = os.path.join(base_dir, hyperparameter_trial['folder'], 'pedestrian_summary.txt')
        tracker_result = pd.read_csv(tracker_results_path, sep=" ")
        tracker_result = tracker_result[METRICS]
        tracker_result['id'] = hyperparameter_trial['id']
        tracker_result['track_thresh'] = hyperparameter_trial['track-thresh']
        tracker_result['track_buffer'] = hyperparameter_trial['track-buffer']
        tracker_result['match_thresh'] = hyperparameter_trial['match-thresh']
        results.append(tracker_result)
    result = pd.concat(results, ignore_index=True).sort_values(by='id')
    return result[['id', 'track_thresh', 'track_buffer', 'match_thresh', *METRICS]]


if __name__ == '__main__':
    base_dir = 'TrackEval/data/trackers/mot_challenge/SportsMOT-val/yolox_tiny_sportsmot-track_results_bytetrack'
    hyperparameter_trials = get_hyperparameter_trials(base_dir)
    result = getResults(base_dir, hyperparameter_trials)
    result.to_csv('outputs/result-hyperparameter-tuning.csv', index=False)
