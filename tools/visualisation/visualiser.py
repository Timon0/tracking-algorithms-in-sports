import argparse
import glob as gb
import json
import os

import cv2
import numpy as np

BASE_PATH = os.getcwd()
VISUAL_BASE_PATH = os.path.join(BASE_PATH, 'visualisation/SportsMOT')

def make_parser():
    parser = argparse.ArgumentParser("Visualiser")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("-gt", default=False)
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-tracker", "--tracker", type=str, default="gt")
    parser.add_argument("-sequence", "--sequence", type=str, default="v_00HRwkvvjtQ_c001")
    return parser

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def generate_images(gt, experiment_name, tracker, split, sequence):
    print("Starting image generation")

    color_list = colormap()
    gt_json_path = os.path.join(BASE_PATH, 'datasets/SportsMOT/annotations', split + '.json')
    img_path = os.path.join(BASE_PATH, 'datasets/SportsMOT', split)

    img_dict = dict()

    if gt:
        txt_path = os.path.join(BASE_PATH, 'datasets/SportsMOT', split, sequence, 'gt/gt.txt')
        visual_path = os.path.join(VISUAL_BASE_PATH, split, "gt", sequence)
        os.makedirs(visual_path, exist_ok=True)
    else:
        txt_path = os.path.join(BASE_PATH, 'outputs', experiment_name, f"track_results_{tracker}", sequence + '.txt')
        visual_path = os.path.join(VISUAL_BASE_PATH, split, experiment_name, tracker, sequence)
        os.makedirs(visual_path, exist_ok=True)

    with open(gt_json_path, 'r') as f:
        gt_json = json.load(f)

    for ann in gt_json["images"]:
        file_name = ann['file_name']
        video_name = file_name.split('/')[0]
        if video_name == sequence:
            img_dict[ann['frame_id']] = os.path.join(img_path, file_name)

    txt_dict = dict()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')

            img_id = linelist[0]
            obj_id = linelist[1]
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]),
                    float(linelist[3]) + float(linelist[5]), int(float(obj_id))]
            if int(img_id) in txt_dict:
                txt_dict[int(img_id)].append(bbox)
            else:
                txt_dict[int(img_id)] = list()
                txt_dict[int(img_id)].append(bbox)

    for img_id in sorted(txt_dict.keys()):
        img = cv2.imread(img_dict[img_id])
        for bbox in txt_dict[img_id]:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          color_list[bbox[4] % 79].tolist(), thickness=2)
            cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color_list[bbox[4] % 79].tolist(), 2)
        cv2.imwrite(visual_path + "/" + sequence + "{:0>6d}.png".format(img_id), img)

    print("Image generation done")


def generate_video_from_images(gt, experiment_name, tracker, split, sequence):
    print("Starting video generation from images")

    if gt:
        visual_path = os.path.join(VISUAL_BASE_PATH, split, "gt", sequence)
    else:
        visual_path = os.path.join(VISUAL_BASE_PATH, split, experiment_name, tracker, sequence)

    img_paths = gb.glob(visual_path + "/*.png")
    fps = 25
    size = (1920, 1080)
    videowriter = cv2.VideoWriter(visual_path + f"/{sequence}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("video generated")


if __name__ == '__main__':
    args = make_parser().parse_args()

    generate_images(args.gt, args.experiment_name, args.tracker, args.split, args.sequence)
    generate_video_from_images(args.gt, args.experiment_name, args.tracker, args.split, args.sequence)
