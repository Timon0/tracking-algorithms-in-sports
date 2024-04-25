import os

SPORTSMOT_DATA_ROOT = "datasets/SportsMOT/splits_txt"

def read_file(file_name):
    with open(file_name, 'r') as file:
        lines = set(file.read().splitlines())
    return lines

def write_file(file_name, lines):
    with open(file_name, 'w') as file:
        for line in lines:
            file.write(line + '\n')


if __name__ == "__main__":
    split = 'val'
    split_file = os.path.join(SPORTSMOT_DATA_ROOT, split + '.txt')
    sequences_from_split = read_file(split_file)

    sports = ['basketball', 'volleyball', 'football']

    for sport in sports:
        sport_file = os.path.join(SPORTSMOT_DATA_ROOT, sport + '.txt')
        sequences_from_sport = read_file(sport_file)

        common_sequences = sequences_from_sport.intersection(sequences_from_split)

        result_file_name = os.path.join(SPORTSMOT_DATA_ROOT, f'{split}-{sport}.txt')
        write_file(result_file_name, common_sequences)
