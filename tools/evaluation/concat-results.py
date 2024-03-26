import pandas as pd
import numpy as np
import sys
import os

trackers = [
    {'folder': 'yolox_x_sportsmot-track_results_sort', 'name': 'SORT (YOLOX-X)', 'average-inference-time': 44.28 },
    {'folder': 'yolox_x_sportsmot-track_results_deepsort', 'name': 'DeepSORT (YOLOX-X)', 'average-inference-time':  71.41 },
    {'folder': 'yolox_x_sportsmot-track_results_bytetrack', 'name': 'ByteTrack (YOLOX-X)', 'average-inference-time':  43.74 },
    {'folder': 'yolox_tiny_sportsmot-track_results_sort', 'name': 'SORT (YOLOX tiny)', 'average-inference-time': 13.04 },
    {'folder': 'yolox_tiny_sportsmot-track_results_deepsort', 'name': 'DeepSORT (YOLOX tiny)', 'average-inference-time': 37.97 },
    {'folder': 'yolox_tiny_sportsmot-track_results_bytetrack', 'name': 'ByteTrack (YOLOX tiny)', 'average-inference-time': 12.20 }
]

def getResults(trackers):
    results = []
    for tracker in trackers:
        tracker_results_path = os.path.join('TrackEval/data/trackers/mot_challenge/SportsMOT-val', tracker['folder'], 'pedestrian_summary.txt')
        tracker_result = pd.read_csv(tracker_results_path, sep=" ")
        tracker_result = tracker_result[METRICS]
        tracker_result['tracker'] = tracker['name']
        tracker_result['FPS'] = round(1000 / tracker['average-inference-time'], 3)
        results.append(tracker_result)
    result = pd.concat(results, ignore_index=True)
    return result[['tracker', *METRICS, 'FPS']]

if __name__ == '__main__':
    result = getResults(trackers)
    result.to_csv('../../outputs/result.csv', index=False)