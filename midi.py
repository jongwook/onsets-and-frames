import sys

import mido
import numpy as np
from tqdm import tqdm


def parse_midi(path):
    """open midi file and return list of (onset, offset, note, velocity)"""
    midi = mido.MidiFile(path)

    time = 0
    notes = []
    for message in midi:
        time += message.time

        if 'note' in message.type:
            note = dict(time=time, note=message.note, velocity=message.velocity)
            notes.append(note)

    rows = []
    for i, note in enumerate(notes):
        if note['velocity'] == 0:
            continue

        onset = note['time']
        offset = [n['time'] for n in notes[i + 1:] if n['note'] == note['note']][0]
        row = onset, offset, note['note'], note['velocity']
        rows.append(row)

    return np.array(rows)


if __name__ == '__main__':
    for input_file in tqdm(sys.argv[1:]):
        if input_file.endswith('.mid'):
            output_file = input_file[:-4] + '.tsv'
        elif input_file.endswith('.midi'):
            output_file = input_file[:-5] + '.tsv'
        else:
            print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
            continue

        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset,offset,note,velocity')
