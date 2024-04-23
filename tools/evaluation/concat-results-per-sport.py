import os

import pandas as pd

METRICS = ['HOTA', 'MOTA', 'IDF1', 'IDSW']
tracker = 'yolox_tiny_sportsmot-track_results_bytetrack'
sports = ['basketball', 'football', 'volleyball']

def getResults(tracker, sports):
    results = []
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
    result = getResults(tracker, sports)
    result.to_csv('outputs/result-per-sports.csv', index=False)
