#! /bin/bash
set -e

echo Downloading the MAESTRO dataset \(87 GB\) ...
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip

echo Extracting the files ...
unzip -o maestro-v1.0.0.zip | awk 'BEGIN{ORS=""} {print "\rExtracting " NR "/2383 ..."; system("")} END {print "\ndone\n"}'

rm maestro-v1.0.0.zip
mv maestro-v1.0.0 MAESTRO

echo Converting the audio files to FLAC ...
COUNTER=0
for f in MAESTRO/*/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/1184) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
done

echo
echo Preparation complete!
