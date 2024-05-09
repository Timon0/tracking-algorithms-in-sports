import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_x_sportsmot")
    parser.add_argument("-tracker", "--tracker", type=str, default="sort")
    parser.add_argument("-subfolder", "--subfolder", type=str, default=None)
    return parser

def main(split, experiment_name, tracker):
    benchmark = f"SportsMOT-{split}"
    tracker_results = f"track_results_{tracker}"
    trackeval_tracker_results = f"{experiment_name}-{tracker_results}"
    TRACKEVAL_TRACKER_ROOT = os.path.join(os.getcwd(), "TrackEval/data/trackers/mot_challenge", benchmark,  trackeval_tracker_results)
    if args.subfolder is not None:
        TRACKEVAL_TRACKER_ROOT = os.path.join(TRACKEVAL_TRACKER_ROOT, args.subfolder)
    TRACKEVAL_TRACKER_DATA_ROOT = os.path.join(TRACKEVAL_TRACKER_ROOT, "data")
    SPORTSMOT_EXPERIMENT_DATA_ROOT = os.path.join(os.getcwd(), "outputs", experiment_name, tracker_results)
    if args.subfolder is not None:
        SPORTSMOT_EXPERIMENT_DATA_ROOT = os.path.join(SPORTSMOT_EXPERIMENT_DATA_ROOT, args.subfolder)

    # ========default hierarchy========
    # folder hierarchy according to
    # https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data
    os.makedirs(TRACKEVAL_TRACKER_ROOT, exist_ok=True)

    # create symlink for data
    os.symlink(SPORTSMOT_EXPERIMENT_DATA_ROOT, TRACKEVAL_TRACKER_DATA_ROOT, target_is_directory=True)

if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args.split, args.experiment_name, args.tracker)