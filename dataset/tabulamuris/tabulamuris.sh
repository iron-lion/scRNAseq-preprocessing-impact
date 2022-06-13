#!/bin/bash
# Original works: https://github.com/hemberg-lab/scRNA.seq.datasets/

mkdir -p ./RAW;
cd ./RAW;
# download data
wget -O FACS.zip \
'https://figshare.com/ndownloader/files/10700143'
unzip FACS.zip

wget -O metadata_FACS.csv 'https://figshare.com/ndownloader/files/10842785'
wget -O annotations_FACS.csv 'https://figshare.com/ndownloader/files/13088129'


wget -O droplet.zip \
'https://figshare.com/ndownloader/files/10700167'
unzip droplet.zip
wget -O metadata_droplet.csv \
'https://figshare.com/ndownloader/files/10700161'
wget -O annotations_droplet.csv \
'https://figshare.com/ndownloader/files/13088039'

