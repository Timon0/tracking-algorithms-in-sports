import os
import shutil

base_dir = 'datasets/SportsMOT'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
train_val_dir = os.path.join(base_dir, 'train_val')

if not os.path.exists(train_val_dir):
    os.makedirs(train_val_dir)

def copy_subfolders(source_dir, destination_dir):
    sequences = []
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(destination_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
            sequences.append(item)
    return sequences

def write_split_txt(train_sequences, val_sequences):
    sequences = train_sequences + val_sequences
    split_file = os.path.join(base_dir, 'splits_txt/train_val.txt')
    with open(split_file, 'w') as file:
        for sequence in sequences:
            file.write(sequence + '\n')


print('Copying sequences to train_val')
train_sequences = copy_subfolders(train_dir, train_val_dir)
print('Copying val sequences to train_val')
val_sequences = copy_subfolders(val_dir, train_val_dir)
print('Write train_val_split.txt file')
write_split_txt(train_sequences, val_sequences)