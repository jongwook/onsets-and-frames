# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.

## Instructions

### Downloading Dataset

The `data` subdirectory already contains the MAPS database. To download the Maestro dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash
python evaluate.py runs/model/model-100000.pt
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/model/model-100000.pt Maestro test
```

## Known issues

The current implementation produces on-par performance on onset predictions, but its frame prediction performance is far worse and is under investigation.

