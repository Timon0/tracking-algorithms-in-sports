import os

import pandas as pd

METRICS = ['HOTA', 'MOTA', 'IDF1', 'IDSW']
trackers = {
    'yolox_tiny_sportsmot_basketball-track_results_bytetrack': ['football', 'volleyball'],
    'yolox_tiny_sportsmot_football-track_results_bytetrack': ['basketball', 'volleyball'],
    'yolox_tiny_sportsmot_volleyball-track_results_bytetrack': ['basketball', 'football'],
}

def getResults(trackers):
    results = []
    for tracker, sports in trackers.items():
        for sport in sports:
            tracker_results_path = os.path.join('TrackEval/data/trackers/mot_challenge/SportsMOT-val', tracker, sport, 'pedestrian_summary.txt')
            tracker_result = pd.read_csv(tracker_results_path, sep=" ")
            tracker_result = tracker_result[METRICS]
            tracker_result['tracker'] = tracker
            tracker_result['sport'] = sport
            results.append(tracker_result)
    result = pd.concat(results, ignore_index=True)
    return result[['tracker', 'sport', *METRICS]]


if __name__ == '__main__':
    result = getResults(trackers)
    result.to_csv('outputs/result-per-sports.csv', index=False)
