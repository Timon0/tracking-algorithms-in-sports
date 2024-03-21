import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TrackEval')))
import trackeval


def make_parser():
    parser = argparse.ArgumentParser("Evaluation")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_x_sportsmot")
    parser.add_argument("-tracker", "--tracker", type=str, default="sort")
    return parser


def eval_mot(**kargs):
    # Command line interface:
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    dataset_config.update(kargs)

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == '__main__':
    args = make_parser().parse_args()

    tracker_results = f"track_results_{args.tracker}"
    trackeval_tracker_results = f"{args.experiment_name}-{tracker_results}"
    GT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'TrackEval/data/gt/mot_challenge'))
    TRACKERS_FOLDER = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'TrackEval/data/trackers/mot_challenge'))

    eval_mot(TRACKERS_TO_EVAL=[trackeval_tracker_results],
             BENCHMARK='SportsMOT',
             SPLIT_TO_EVAL=args.split,
             TRACKERS_FOLDER=TRACKERS_FOLDER,
             GT_FOLDER=GT_FOLDER,
             GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt')
