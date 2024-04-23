import json
import os
import numpy as np


def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def generate_images_list(split):
    annotation_file = os.path.join('datasets/SportsMOT/annotations', split + ".json")
    with open(annotation_file, 'r') as file:
        data = json.load(file)
    filenames_from_file = [image["file_name"] for image in data["images"]]

    export_file = os.path.join('datasets/SportsMOT/', split, split + "-images.txt")
    with open(export_file, 'w') as file:
        for filename in filenames_from_file:
            file.write(f'{filename}\n')


def generate_labels_per_frame(split):
    seq_root = os.path.join('datasets/SportsMOT', split)
    label_root = os.path.join('datasets/SportsMOT', split)
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1

    for seq in seqs:
        seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = os.path.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if mark == 0 or not label == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = os.path.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


if __name__ == "__main__":
    generate_labels_per_frame('train')
    generate_images_list('train')

    generate_labels_per_frame('val')
    generate_images_list('val')
